from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")

chat = [
    { "role": "user", "content": "hey can you do that"},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(prompt)
