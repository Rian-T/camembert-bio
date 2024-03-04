import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from meditron_benchmark_test import benchmark_factory, load_instruction

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

benchmarks = ["pubmedqa", "medmcqa", "medqa4", "blurb", "mmlu_medical", "gsm8k"]

metrics_per_benchmark = {}

torch.cuda.empty_cache()

model_id = "meta-llama/Llama-2-7b-chat-hf"
sampling_params = SamplingParams(temperature=0.1, top_p=0.95)
llm = LLM(model_id, tensor_parallel_size=2, dtype="half")
tokenizer = AutoTokenizer.from_pretrained(model_id)

for benchmark in benchmarks:
    logfile_name = model_id.split("/")[0] + "_" + benchmark + "_log.txt"
    with open(logfile_name, "w") as benchmark_log_file:
        pubmedqa = benchmark_factory(benchmark)
        pubmedqa.load_data("test")
        pubmedqa.load_data("train")
        pubmedqa.preprocessing("test")
        pubmedqa.preprocessing("train")
        pubmedqa.add_instruction(load_instruction(benchmark), partition="train")
        pubmedqa.add_instruction(load_instruction(benchmark), partition="test")
        pubmedqa.add_few_shot(shots=3)

        def parse_answer(answer):
            answer = answer.lower()
            if "yes" in answer:
                return "yes"
            elif "no" in answer:
                return "no"
            elif "maybe" in answer:
                return "maybe"
            else:
                return "unknown"

        def generate_prompts(texts):
            def chat_to_prompt(text):
                chat = [
                    { "role": "user", "content": text},
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                return prompt
            if model_id in ["meta-llama/Llama-2-7b-chat-hf"]:
                prompts = [chat_to_prompt(text) for text in texts]
            else:
                prompts = texts
            return prompts

        test_data = pubmedqa.test_data.shuffle()
        golds = test_data["gold"]

        prompts = generate_prompts(test_data["prompt"])
        outputs = llm.generate(prompts, sampling_params)

        preds = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            pred = parse_answer(generated_text)
            preds.append(pred)

        print("F1:", f1_score(golds, preds, average="micro"))
        print("Precision:", precision_score(golds, preds, average="micro"))
        print("Recall:", recall_score(golds, preds, average="micro"))
        print("Accuracy:", accuracy_score(golds, preds))

        metrics = {
            "f1": f1_score(golds, preds, average="micro"),
            "precision": precision_score(golds, preds, average="micro"),
            "recall": recall_score(golds, preds, average="micro"),
            "accuracy": accuracy_score(golds, preds),
        }

        print(metrics)
        
        metrics_per_benchmark[benchmark] = metrics
        print(metrics_per_benchmark)
        benchmark_log_file.write(str(metrics))