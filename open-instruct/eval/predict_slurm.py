
'''
This script is used to get models' predictions on a set of prompts (put in files with *.jsonl format, 
with the prompt in a `prompt` field or the conversation history in a `messages` field).

For example, to get predictions on a set of prompts, you should put them in a file with the following format:
    {"id": <uniq_id>, "prompt": "Plan a trip to Paris."}
    ...
Or you can use the messages format:
    {"id": <uniq_id>, "messages": [{"role": "user", "content": "Plan a trip to Paris."}]}
    ...

Then you can run this script with the following command:
    python eval/predict.py \
        --model_name_or_path <huggingface_model_name_or_path> \
        --input_files <input_file_1> <input_file_2> ... \
        --output_file <output_file> \
        --batch_size <batch_size> \
        --use_vllm
'''

def sprint_verbosity(verbose):
    def inner(txt):
        if verbose:
            print(f"SBATCH_INFO: {txt}")
    return inner


import argparse
import json
import os
import importlib
import vllm
import torch
import sys
from pathlib import Path
# force add parent of parent of directory of script to path (open-instruct)
sys.path.append(str(Path(__file__).parent.parent.resolve()))
print(f"{sys.path=}")

from eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model, dynamic_import_function

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Huggingface model name or path.")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Huggingface tokenizer name or path."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str,
        help="OpenAI engine name. This should be exclusive with `model_name_or_path`.")
    parser.add_argument(
        "--input_files", 
        type=str, 
        nargs="+",
        help="Input .jsonl files, with each line containing `id` and `prompt` or `messages`.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/model_outputs.jsonl",
        help="Output .jsonl file, with each line containing `id`, `prompt` or `messages`, and `output`.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for prediction.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument(
        "--load_in_float16",
        action="store_true",
        help="By default, huggingface model will be loaded in the torch.dtype specificed in its model_config file."
             "If specified, the model dtype will be converted to float16 using `model.half()`.")
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput.")
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="maximum number of new tokens to generate.")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="whether to use sampling ; use greedy decoding otherwise.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for sampling.")
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p for sampling.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Whether to print additional info (labelled SBATCH_INFO)")
    parser.add_argument(
        "--stop_token",
        type=str,
        help="Stop token to use for SamplingParams")
    args = parser.parse_args()

    # model_name_or_path and openai_engine should be exclusive.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "model_name_or_path and openai_engine should be exclusive."
    return args

def dynamic_import_fn(import_path):
    """Imports a function from eval/templates.py via importlib to resolve name conflicts.
    The folder "eval" conflicts with the builtin function `eval`"""

    splitted = import_path.rsplit(".", 1)
    module_name, fn_name = splitted
    assert module_name == "eval.templates"

    # The current script is in the eval folder
    module_path = Path(__file__).parent / "templates.py"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    templates = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = templates
    spec.loader.exec_module(templates)
    return getattr(templates, fn_name)

if __name__ == "__main__":
    args = parse_args()
    sprint = sprint_verbosity(args.verbose)

    # check if output directory exists
    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    sprint("Checked output dir")

    # load the data
    for input_file in args.input_files:
        with open(input_file, "r") as f:
            instances = [json.loads(x) for x in f.readlines()]
    sprint("Loaded data from jsonl")

    if args.model_name_or_path is not None:
        prompts = []
        chat_formatting_function = dynamic_import_fn(args.chat_formatting_function) if args.use_chat_format else None
        for instance in instances:
            if "messages" in instance:
                if not args.use_chat_format:
                    raise ValueError("If `messages` is in the instance, `use_chat_format` should be True.")
                assert all("role" in message and "content" in message for message in instance["messages"]), \
                    "Each message should have a `role` and a `content` field."
                prompt = chat_formatting_function(instance["messages"], tokenizer=None, add_bos=False)
            elif "prompt" in instance:
                if args.use_chat_format:
                    messages = [{"role": "user", "content": instance["prompt"]}]
                    prompt = chat_formatting_function(messages, add_bos=False)
                else:
                    prompt = instance["prompt"]
            else:
                raise ValueError("Either `messages` or `prompt` should be in the instance.")
            prompts.append(prompt)
        if args.use_vllm:
            sprint("Creating vllm model")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
            )
            sprint("Instantiating vllm SamplingParams")
            if args.stop_token:
                sampling_params = vllm.SamplingParams(
                    temperature=args.temperature if args.do_sample else 0, 
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    stop=args.stop_token
                )
            else:
                sampling_params = vllm.SamplingParams(
                    temperature=args.temperature if args.do_sample else 0, 
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens
                )
            sprint("Generating results!")
            outputs = model.generate(prompts, sampling_params)
            sprint("Finished generating results")
            outputs = [it.outputs[0].text for it in outputs]
            sprint(f"{outputs[:2]=}")
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        sprint("Writing to file")
        with open(args.output_file, "w") as f:
            for instance, output in zip(instances, outputs):
                instance["output"] = output
                f.write(json.dumps(instance) + "\n")
        sprint("Finished writing to file")
                
    elif args.openai_engine is not None:
        query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            output_path=args.output_file,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
    else:
        raise ValueError("Either model_name_or_path or openai_engine should be provided.")

    print("Done.")
