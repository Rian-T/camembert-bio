import argparse
import gc

import wandb
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from meditron_benchmark_test import benchmark_factory, load_instruction
from langchain.retrievers import PubMedRetriever
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

ADD_RAG = False
BENCHMARKS = ["pubmedqa" , "medqa4", "blurb", "mmlu_medical", "gsm8k"]
MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b-it",
    "google/gemma-7b",
    "google/gemma-2b-it",
    "google/gemma-2b",
    "BioMistral/BioMistral-7B",
    "epfl-llm/meditron-7b",
    "croissantllm/CroissantLLMBase",
    "croissantllm/CroissantLLMChat-v0.1"
]
METRICS_PER_BENCHMARK = {}

torch.cuda.empty_cache()
retriever = PubMedRetriever()


def get_context(query, n=2, truncation=1000):
    """Retrieve relevant documents for a given query."""
    try:
        results = retriever.get_relevant_documents(query)
        results_content = " ".join([result.page_content for result in results])
        return "\nAdditional knowledge: " + results_content[:truncation] + "\n"
    except:
        return ""


def find_relevant_context(prompt):
    """Find the relevant context in a given prompt."""
    context = prompt.split("\n")[-3]
    return get_context(context)


def insert_text_before_last_line(original_text, line_to_find, text_to_insert):
    """Insert a text before the last occurrence of a specific line in a string."""
    lines = original_text.split("\n")
    for i in reversed(range(len(lines))):
        if lines[i].strip() == line_to_find:
            lines.insert(i, text_to_insert)
            break
    return "\n".join(lines)


def insert_text_after_line(original_text, line_to_find, text_to_insert):
    """Insert a text after a specific line in a string."""
    lines = original_text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == line_to_find:
            lines.insert(i + 1, text_to_insert)
            break
    return "\n".join(lines)


def parse_answer(answer):
    """Parse the answer from the generated text."""
    # answer = answer.lower()
    # if "yes" in answer:
    #     return "yes"
    # elif "no" in answer:
    #     return "no"
    # elif "maybe" in answer:
    #     return "maybe"
    # else:
    #     return "unknown"
    # return answer between "The answer is:" and the next line if there is one
    if "The answer is:" in answer:
        answer = answer.split("The answer is:")[1].strip()
        if "\n" in answer:
            answer = answer.split("\n")[0]
        return answer.strip()
    else:
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
        if "." in answer:
            answer = answer.split(".")[0].strip()
        return answer

def generate_prompts(texts, tokenizer, model_id, add_rag):
    """Generate prompts for the LLM model."""
    if add_rag:
        for i, text in enumerate(tqdm(texts)):
            context = find_relevant_context(text)
            texts[i] = insert_text_before_last_line(text, "The answer is:", context)
            print(texts[i])

    def chat_to_prompt(text):
        chat = [
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        return prompt

    if model_id in [
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/gemma-7b-it",
        "google/gemma-2b-it",
        "croissantllm/CroissantLLMChat-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]:
        prompts = [chat_to_prompt(text) for text in texts]
    else:
        prompts = texts
    return prompts


def run_benchmark(benchmark, llm, tokenizer, sampling_params, model_id, add_rag):
    """Run the benchmark for a given model."""
    logfile_name = (
        f"{model_id.split('/')[1]}_{benchmark}_{'rag' if add_rag else ''}_log.txt"
    )
    with open(logfile_name, "a") as benchmark_log_file:
        benchmark_test = benchmark_factory(benchmark)
        benchmark_test.load_data("train")
        benchmark_test.load_data("test")
        benchmark_test.preprocessing("train")
        benchmark_test.preprocessing("test")
        benchmark_test.add_instruction(load_instruction(benchmark if not "medqa4" else "medqa"), partition="train")
        benchmark_test.add_instruction(load_instruction(benchmark if not "medqa4" else "medqa"), partition="test")
        benchmark_test.add_few_shot(shots=2)

        test_data = benchmark_test.test_data.shuffle()
        golds = test_data["gold"]
        prompts = test_data["prompt"]

        prompts = generate_prompts(prompts, tokenizer, model_id, add_rag)
        outputs = llm.generate(prompts, sampling_params)
        preds = [parse_answer(output.outputs[0].text) for output in outputs]
        
        preds = ["unknown" if pred is None else pred for pred in preds]
        golds = ["none" if gold is None else gold for gold in golds]

        metrics = {
            # "f1": f1_score(golds, preds, average="micro"),
            # "precision": precision_score(golds, preds, average="micro"),
            # "recall": recall_score(golds, preds, average="micro"),
            "accuracy": accuracy_score(golds, preds),
        }

        wandb.log({f"{benchmark}/{model_id.split('/')[1]}_{'rag' if add_rag  else ''}": value for key, value in metrics.items()})

        print(metrics)
        METRICS_PER_BENCHMARK[benchmark] = metrics
        print(METRICS_PER_BENCHMARK)
        benchmark_log_file.write(str(metrics))




def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run LLM QA benchmark.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='The size of tensor parallelism.')
    parser.add_argument('--additional_models', nargs='*', default=[],
                        help='Additional models to append to the existing list.')
    parser.add_argument('--dtype', type=str, default='half',
                        help='The data type to use for the model. Options: "half", "float".')
    args = parser.parse_args()

    # Append additional models to the existing list
    models = args.additional_models + MODELS 

    wandb.init(project="llm_bioqa", name="run_llm_rag", entity="rntc")
    for add_rag in [False, True]:
        for model_id in models:  # Use the updated models list
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=12)
            llm = LLM(model_id, tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            for benchmark in BENCHMARKS:
                try:
                    run_benchmark(benchmark, llm, tokenizer, sampling_params, model_id, add_rag)
                except Exception as e:
                    print(e)
            destroy_model_parallel()
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
