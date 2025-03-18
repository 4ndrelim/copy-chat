## Finetuning Simplified

This folder has been adapted to simplify the finetuning process and made easily acessible to anyone. Here are the steps:


## 0, Installation
Note: if nvcc is not yet installed, do `conda install nvidia/label/cuda-12.1.1::cuda-toolkit`

You can test it by running `nvcc --version`

```
pip install --upgrade pip "setuptools<70.0.0" wheel 
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt
```

## 1. Download The Correct Model + Tokenizer
See [here](../download/README.md).

## 2. Prepare Dataset For Training
Your only involvement in writing code is likely here. To finetune, ensure dataset is of the following format:

```json
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},  // 'generated response'
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},  // 'generated response'
        ...
    ],
}
```

You can see a sample [here](./datasets/formatted_datasets/prepared_tweet.jsonl) too.

### Chat Template
You can specify the system, user, and assistant prompt(s) [here](./dataset_preparation/prompt_templates/).


## 3. Specify Config File
You can specify the training config in [`scripts/custom_scripts`](./scripts/custom_scripts/)

## 4. Run!
Sample command
```bash
sh scripts/custom_scripts/tweet_lora
```

### Merging the LoRA Adaptors
The above command will save the LoRA adaptors. To merge with the base model, run the merging script.

**NOTE**: You might want to specify the base model and adaptors path in the script.

```bash
sh scripts/custom_scripts/merge_lora.sh
```

## 5. [Optional] Prediction
To perform inference, the model first needs to be hosted. Navigate to the directory where the the vllm folder resides. Below is a sample command.

Tip: Make sure you're in the right conda env for deployment!

```bash
python -m vllm.entrypoints.openai.api_server --load-format safetensors --dtype bfloat16 --max-model-len 2048 --tensor-parallel-size 1 --model custom_model_path
```
