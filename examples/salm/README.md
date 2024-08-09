
## Setting up the environment
This code needs to be executed with both Nemo and TRT-LLM support. This container can be used to execute it:

gitlab-master.nvidia.com:5005/dl/joc/nemo-ci/trtllm_0.11/train:pipe.16718524-x86

## Obtain datasets

Download wav file

```
wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
```

## Running the inference

```
# it has to contain $engines_dir/encoder and $engines_dir/decoder
engines_dir=./trt_engines 

# this is the global config_file (including all components)
config_file=./nemo_model/config.yaml 

# tokenizer model file
tokenizer_path=./nemo_model/XXXXXX_tokenizer.model 

# wav file from assets
wav_file=assets/1221-135766-0002.wav

```

Execute the script to run the inference on one wav file:

```
python3 run.py --name single_wav_test --engine_dir  $engines_dir --config_path $config_file --tokenizer_path $tokenizer_path --input_file $assets --batch_size 1
```
