## Finetuning Simplified

This folder has been adapted to simplify the finetuning process and made easily acessible to anyone. Here are the steps:


## 0, Installation
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
Dataset needs to be processed to be of the following format:

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
### Raw Dataset
Expected to be of JSONL format.

### Chat Template
You can specify the system, user, and assistant prompt(s) [here](./dataset_preparation/prompt_templates/).

### Dataset Converter
Your only involvement in writing code is likely [here](./dataset_preparation/dataset_preparer.py) in `./dataset_preparation/dataset_preparer.py`. Code the logic for extracting relevant fields from raw dataset.

### Conversion From Raw Datasets
Below is a sample command to transform raw datasets into the format above:

```bash
python -m dataset_preparation.dataset_preparer \ 
--dataset_name example_dataset \ 
--input_path data/example/train.jsonl \
--template_path dataset_preparation/prompt_templates/example.json
```

## 3. Specify Config File
You can specify the training config in [`scripts/custom_scripts`](./scripts/custom_scripts/)

## 4. Run!
Sample command
```bash
sh scripts/custom_scripts/books_lora.sh
```

### Merging the LoRA Adaptors
The above command will save the LoRA adaptors. To merge with the base model, run the merging script.

**NOTE**: You might want to specify the base model and adaptors path in the script.

```bash
sh scripts/custom_scripts/merge_lora.sh
```

## 5. [Optional] Prediction
To perform inference, the model first needs to be hosted. Navigating to the directory where the saved model is stored, below is a sample command:

```bash
python -m vllm.entrypoints.openai.api_server --load-format safetensors --dtype bfloat16 --max-model-len 2048 --tensor-parallel-size 1 --model custom_model_path
```

### Prediction
Below is a sample command:
```bash
python -m autothought_eval.predict \ 
--model example_model \ 
--prompts_template_path dataset_preparation/prompt_templates/example.json \
--input_file data/example/test.jsonl
```

