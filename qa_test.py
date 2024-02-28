import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from meditron_benchmark_test import benchmark_factory, load_instruction

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

benchmarks = ["pubmedqa", "medmcqa", "medqa4", "blurb", "mmlu_medical", "gsm8k"]

metrics_per_benchmark = {}

torch.cuda.empty_cache()

for benchmark in benchmarks:
    with open(f"{benchmark}_log.txt", "w") as benchmark_log_file:
        pubmedqa = benchmark_factory(benchmark)
        pubmedqa.load_data("test")
        pubmedqa.load_data("train")
        pubmedqa.preprocessing("test")
        pubmedqa.preprocessing("train")
        pubmedqa.add_instruction(load_instruction(benchmark), partition="train")
        pubmedqa.add_instruction(load_instruction(benchmark), partition="test")
        pubmedqa.add_few_shot(shots=3)

        models = ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-7b-it", "BioMistral/BioMistral-7B", "epfl-llm/meditron-7b", "croissantllm/CroissantLLMChat-v0.1"]
        metrics_per_model = {}
        for model_id in models:
            #model_id = "meta-llama/Llama-2-7b-chat-hf"
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            # )

            sampling_params = SamplingParams(temperature=0.1, top_p=0.95)
            llm = LLM(model_id, tensor_parallel_size=1, dtype="half")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_id,
            #     quantization_config=bnb_config,
            #     device_map={"": 0},
            # )

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
                return [chat_to_prompt(text) for text in texts]

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

            metrics_per_model[model_id] = {
                "f1": f1_score(golds, preds, average="micro"),
                "precision": precision_score(golds, preds, average="micro"),
                "recall": recall_score(golds, preds, average="micro"),
                "accuracy": accuracy_score(golds, preds),
            }

            print(metrics_per_model)
        
        metrics_per_benchmark[benchmark] = metrics_per_model
        print(metrics_per_benchmark)
        benchmark_log_file.write(str(metrics_per_model))
