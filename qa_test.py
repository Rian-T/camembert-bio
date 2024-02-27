import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from meditron_benchmark_test import benchmark_factory, load_instruction

pubmedqa = benchmark_factory("pubmedqa")
pubmedqa.load_data("test")
pubmedqa.load_data("train")
pubmedqa.preprocessing("test")
pubmedqa.preprocessing("train")
pubmedqa.add_few_shot(shots=3)
pubmedqa.add_instruction(load_instruction("pubmedqa"), partition="test")

model_id = "meta-llama/Llama-2-7b-chat-hf"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model_id, tensor_parallel_size=1)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map={"": 0},
# )

text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = llm.generate(text, max_new_tokens=20)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(outputs)

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


# def generate(text):
#     text = "<s> [INST] " + text + " [/INST]"
#     inputs = tokenizer(text, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_new_tokens=6)
#     answer = (
#         tokenizer.decode(outputs[0], skip_special_tokens=True)
#         .split("[/INST]")[1]
#         .strip()
#     )
#     print(answer)
#     return answer


test_data = pubmedqa.test_data.shuffle()
golds = test_data["gold"]
#preds = [parse_answer(generate(prompt)) for prompt in test_data["prompt"]]

print(golds)
#print(preds)

# compute f1 score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# print("F1:", f1_score(golds, preds, average="micro"))
# print("Precision:", precision_score(golds, preds, average="micro"))
# print("Recall:", recall_score(golds, preds, average="micro"))
# print("Accuracy:", accuracy_score(golds, preds))
