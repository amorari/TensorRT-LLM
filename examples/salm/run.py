# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import re
import time
import yaml
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from whisper.normalizers import EnglishTextNormalizer
from whisper_utils import (N_SAMPLES, pad_or_trim,
                           store_transcripts, write_error_stats)
from whisper_utils import load_audio, load_audio_wav_format, pad_or_trim, N_SAMPLES

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo

from transformers import AutoTokenizer

from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from nemo.collections.multimodal.speech_llm.modules.perception_modules import AudioPerceptionModule



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='warning')
    parser.add_argument('--engine_dir', type=str, default='trt_engine')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='tmp')
    parser.add_argument('--assets_dir', type=str, default='./assets')
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--dataset',
                        type=str,
                        default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument('--name',
                        type=str,
                        default="librispeech_dummy_benchmark")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable_warmup', action='store_true')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16'])
    parser.add_argument('--accuracy_check',
                        action='store_true',
                        help="only for CI test")
    return parser.parse_args()


def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=0):
    if input_tensor.dim() == 2:
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    elif input_tensor.dim() == 3:
        # Audio tensor case: batch, seq_len, feature_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"
        batch_size, seq_len, feature_len = input_tensor.shape

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(batch_size):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length, :])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return output_tensor


def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_dtype_from_json_config(precision):
    if 'fp16' in precision:
        dtype = 'float16'
    elif 'bf16' in precision:
        dtype = 'bfloat16' 
    elif 'fp32' in precision:
        dtype = 'float32'
    else:
        raise ValueError(f"Unsupported precision {precision}")
    return dtype

class SalmPreprocessor:
    
        def __init__(self, config, device='cuda'):
            self.sample_rate = config['sample_rate']

            # TODO: Reimplement to avoid NeMo dependency
            self.preprocessor = AudioToMelSpectrogramPreprocessor(
                dither=config['dither'],
                features=config['features'],
                frame_splicing=config['frame_splicing'],
                n_fft=config['n_fft'],
                normalize=config['normalize'],
                pad_to=config['pad_to'],
                sample_rate=config['sample_rate'],
                window=config['window'],
                window_size=config['window_size'],
                window_stride=config['window_stride']
            )

            self.preprocessor.to(device)
            self.device = device
    
        def __call__(self, audio):
            return self.preprocessor(audio)

        def log_mel_spectrogram_from_file(self, file_path):
            if file_path.endswith('.wav'):
                audio, _ = load_audio_wav_format(file_path)
            else:
                audio = load_audio(file_path)
            lengths = torch.tensor([audio.shape[0]])
            audio = torch.from_numpy(audio).unsqueeze(0)  # Convert to tensor and add batch dimension

            return self.log_mel_spectrogram(audio, lengths)

        def log_mel_spectrogram(
                self,
                audio: torch.Tensor = None,
                lengths: torch.Tensor = None,
            ):

            if not audio.dim() == 2:
                raise ValueError("audio must have 2 dimensions: batch, seq_len")

            print(f"audio ({type(audio)}) shape: {audio.shape}")
            print(f"lengths ({type(lengths)}): {lengths}")

            if self.device is not None:
                audio = audio.to(self.device)
                lengths = lengths.to(self.device)


            processed_signal, processed_signal_length = self.preprocessor.get_features(audio, lengths)
            return processed_signal, processed_signal_length


class SalmEncoding:

    def __init__(self, engine_dir, config_encoder, config_modality_adapter, 
                 encoder_input_dtype='float32',
                 ecnoder_input_length_dtype='int64'):
        self.session = self.get_session(engine_dir)
        self.config_encoder = config_encoder
        self.config_modality_adapter = config_modality_adapter
        self.n_mels = config_encoder['feat_in']
        self.encoder_input_dtype = encoder_input_dtype
        self.encoder_input_length_dtype = ecnoder_input_length_dtype

    def get_session(self, engine_dir):
        serialize_path = engine_dir / 'encoder' / 'rank0.trt'
        with open(serialize_path, 'rb') as f:
           session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self, processed_signal, processed_signal_length):
   
        inputs = OrderedDict()
        inputs['processed_signal'] = processed_signal
        inputs['processed_signal_length'] = processed_signal_length
        output_list = [
            TensorInfo('processed_signal', str_dtype_to_trt(self.encoder_input_dtype),
                       processed_signal.shape),
            TensorInfo('processed_signal_length', str_dtype_to_trt(self.encoder_input_length_dtype),
                       processed_signal_length.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        print(f"outputs: {outputs.keys()}")
        encoded = outputs['encoded']
        encoded_len = outputs['encoded_len']
        print(f"encoder_output shape: {encoded.shape}")
        print(f"encoder_output_length: {encoded_len}")
         
        return encoded, encoded_len


class SalmDecoding:

    def __init__(self, engine_dir, runtime_mapping, config_decoder, debug_mode=False):

        self.config_decoder = config_decoder
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)
        

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            paged_kv_cache=self.decoder_config['plugin_config']
            ['paged_kv_cache'],
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_max_input_length,
                 encoder_input_lengths,
                 eot_id,
                 max_new_tokens=40,
                 num_beams=1):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones(
            [batch_size, 1, encoder_max_input_length]).int().cuda()

        # generation config
        sampling_config = SamplingConfig(end_id=eot_id,
                                         pad_id=eot_id,
                                         num_beams=num_beams)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length)

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID)
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full((encoder_outputs.shape[0], ),
                                                 encoder_outputs.shape[1],
                                                 dtype=torch.int32,
                                                 device='cuda')

                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_output_lens)
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class SalmTRTLLM(object):

    def __init__(self, engine_dir, tokenizer_path, config=None):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        dtype = get_dtype_from_json_config(config['precision'])


        self.encoder = SalmEncoding(engine_dir,
                                    config['perception']['encoder'],
                                    config['perception']['modality_adapter'])

        '''
        TODO: this is not ready
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  
        self.decoder = SalmDecoding(engine_dir,
                                        tokenizer_path,
                                        runtime_mapping,
                                        debug_mode=False)
        '''

    def process_batch(
            self,
            processed_signal,
            processed_signal_length,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            num_beams=1):
        
        encoded, encoded_len = self.encoder.get_audio_features(processed_signal, processed_signal_length)
        print(f"encoded shape: {encoded.shape}")
        print(f"encoded_len: {encoded_len}")

        '''
        TODO: to complete
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id)
        batch_size = processed_signal.shape[0]

        decoder_input_ids = prompt_id.repeat(batch_size, 1)
        output_ids = self.decoder.generate(decoder_input_ids,
                                           encoder_output,
                                           encoder_max_input_length,
                                           self.eot_id,
                                           max_new_tokens=96,
                                           num_beams=num_beams)
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts
        '''
        return []


def decode_wav_file(
        input_file_path,
        model,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        batch_size=1,
        num_beams=1,
        normalizer=None,
        preprocessor=None):

    processed_signal, processed_signal_lenght = preprocessor.log_mel_spectrogram_from_file(
                                            file_path= input_file_path) 

    print(f"processed_signal shape: {processed_signal.shape}")
    print(f"total_duration: {processed_signal_lenght}")
    # repeat the processed_signal spectrogram to match the batch size
    processed_signal = processed_signal.repeat(batch_size, 1, 1)
    predictions = model.process_batch(processed_signal, processed_signal_lenght, text_prefix, num_beams)
    print(predictions)
    prediction = predictions[0]

    prediction = re.sub(r'<\|.*?\|>', '', prediction)
    if normalizer:
        prediction = normalizer(prediction)
    print(f"prediction: {prediction}")
    results = [(0, [""], prediction.split())]
    return results, processed_signal_lenght


def collate_wrapper(batch):
    speeches, durations, labels, ids = [], [], [], []
    for item in batch:
        speech = item["audio"]["array"]
        duration = speech.shape[-1]
        speech = pad_or_trim(speech, N_SAMPLES)
        speech = speech.astype(np.float32)
        speech = torch.from_numpy(speech)
        speeches.append(speech)
        durations.append(duration)
        labels.append(item["text"])
        ids.append(item["id"])
    return speeches, durations, labels, ids


def decode_dataset(
        model,
        dataset,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        batch_size=1,
        num_beams=1,
        normalizer=None,
        sample_rate=16000,
        preprocessor = None):
    librispeech_dummy = load_dataset(dataset, "clean", split="validation")

    data_loader = DataLoader(librispeech_dummy,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True,
                             collate_fn=collate_wrapper)
    results = []
    total_duration = 0



    for batch in data_loader:
        waveforms, lengths, texts, ids = batch
        total_duration += sum(lengths) / sample_rate

        for wave in waveforms:
            assert wave.is_pinned()

        processed_signal, processed_signal_length = [
            preprocessor.log_mel_spectrogram(audio=wave,
                                lenghts=lengths).unsqueeze(0)
            for wave in waveforms
        ]

        predictions = model.process_batch(processed_signal, processed_signal_length, text_prefix, num_beams)
        for wav_id, label, prediction in zip(ids, texts, predictions):
            # remove all special tokens in the prediction
            prediction = re.sub(r'<\|.*?\|>', '', prediction)
            if normalizer:
                prediction, label = normalizer(prediction), normalizer(label)
            print(f"wav_id: {wav_id}, label: {label}, prediction: {prediction}")
            results.append((wav_id, label.split(), prediction.split()))
    return results, total_duration


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    config = read_config(Path(args.config_path))

    batch_size = args.batch_size

    model = SalmTRTLLM(args.engine_dir, args.tokenizer_path, config=config)

    preprocessor = SalmPreprocessor(config['perception']['preprocessor'], device='cuda')

    normalizer = EnglishTextNormalizer()
    if args.enable_warmup:
        results, total_duration = decode_dataset(
            model,
            "hf-internal-testing/librispeech_asr_dummy",
            batch_size=batch_size,
            num_beams=args.num_beams,
            normalizer=normalizer,
            preprocessor=preprocessor,
            mel_filters_dir=args.assets_dir)
    start_time = time.time()
    if args.input_file:
        results, total_duration = decode_wav_file(
            args.input_file,
            model,
            batch_size=batch_size,
            num_beams=args.num_beams,
            preprocessor=preprocessor)
    else:
        results, total_duration = decode_dataset(
            model,
            args.dataset,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            normalizer=normalizer,
            preprocessor=preprocessor)
    elapsed = time.time() - start_time
    results = sorted(results)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    store_transcripts(filename=f"{args.results_dir}/recogs-{args.name}.txt",
                      texts=results)

    with open(f"{args.results_dir}/errs-{args.name}.txt", "w") as f:
        total_error_rate = write_error_stats(f,
                                             "test-set",
                                             results,
                                             enable_log=True)
        if args.accuracy_check and args.dataset == "hf-internal-testing/librispeech_asr_dummy" and not args.input_file:
            assert total_error_rate <= 2.8, f"Word Error rate using whisper large-v3 model should be 2.40%, but got {total_error_rate}"

    rtf = elapsed / total_duration
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    s += f"num_beams: {args.num_beams}\n"
    s += f"total error rate: {total_error_rate:.2f}%\n"
    print(s)

    with open(f"{args.results_dir}/rtf-{args.name}.txt", "w") as f:
        f.write(s)

    del model
