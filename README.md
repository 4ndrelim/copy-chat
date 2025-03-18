# High-Level Overview
This guide simplifies and streamlines the steps for deployment and finetuning.

## Tips
Always ensure you are in the right conda environment before executing. The scripts have been set-up such that different tasks will use different conda env. This is a safeguard against the very dynamic landscape of ML where dependency conflicts happen far too often for anyone's liking. 

To see the list of environments:

`conda env list`

To create and enter an environment:

`conda create --name <name> python=3.10 && conda activate <name>`

## Data Ingestion Pipeline
TODO

## Downloading the Models
note: You can do this in the (base) conda env, which is the default.
1. cd download/
2. In download.sh, specify:
    1. Your Hugging Face token export HF_TOKEN="...".
    2. The models you wish to download from Hugging Face.
3. ./download.sh
    1. You can do this directly in the terminal, but it is strongly advised to do it in tmux because some downloads can take hours.
    2. To create a new session, tmux new-session -s <my_session>.
    3. To attach to an existing session, tmux attach-session -t <my_session>.
    4. To kill a session tmux kill-session -t <my_session>.
4. Models are saved to /home/llm/models.

## Finetuning Script 
We will be using a customised version of open-instruct. See [here](./open-instruct/).

## Deployment
Here we leverage vLLM for its superb KV cache support alongside superb batching capability, leading to high throughput generation.
0. `conda create --name vllm-deploy python=3.10`
1. `conda activate vllm-deploy`
2. Modify and run `./deploy.sh`

There are plenty parameters you can configure deploy.sh, but here are the common ones,

1. MODEL_NAME and TOKENIZER should point to the same path - the directory where the model was downloaded into
    - Unless you change / wish to use a different tokenizer.
    - mdoels are stored at /home/llm/models.
2. SEVRED_NAME is the name of your deployed model, which will be referenced in the payload of the information sent to vLLM.
3. PRECISION_TYPE is the size of the parameter. Take note of memory requirements.
    - for a model of size 32B (32 billion params), FP16 (2 bits per param) would imply 64GB memory requirement.
    - In general, BF16 is sufficient for inference but FP32 (or BF16) is preferred for finetuning
    - In environments where BF16 is not supported, FP16 should suffice for inference, but you **may** get marginally better results by deploying the model with FP32 (assuming there's sufficient memory capacity) 
    - If desperate for memory, it's worth trying INT8
4. MAX_LEN is a cap on both the number of input and output tokens **combined**.
    - Note that this value must not exceed the context window of the model.
    - in practice, your input prompt and output response tokens combined should not exceed 80% of the context window length, which would otherwise face degrading capabilities (the last few tokens of the output)
5. CUDA_VISIBLE_DEVICES specifies which GPUs are available for your session
    - Useful when more than 1 users or different GPUs for different tasks
6. MODEL_PARALLELISM splits the mdoel weights across the GPU
    - Using the previous motivating example of 32B model on FP16, setting this value to be 4 would mean 64GB is split across 4 GPUs - 16GB each
    - To allow some memory for KV cache during vLLM deployment, try not to exhaust past ~85% of available memory.
7. PORT specifies the port you want the endpoint to use.
3. ./deploy.sh
