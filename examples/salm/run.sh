#!/bin/bash
#export TRTTLM_PD_PROFILE_START_STOP=
output_dir=salm_engine
python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/1221-135766-0002.wav
#python3 run.py --engine_dir salm_engine --dataset hf-internal-testing/librispeech_asr_dummy --batch_size 1



