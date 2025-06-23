from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "rhaymison/Llama-3-portuguese-Tom-cat-8b-instruct"

# Carrega tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Configura pipeline com parâmetros mais seguros
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    do_sample=True,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

conversation = []

while True:
    user = input("> ")
    conversation.append(f"User: {user}")
    prompt = "\n".join(conversation) + "\nAssistant:"
    
    out = pipe(prompt)
    generated_text = out[0]["generated_text"]
    resp = generated_text[len(prompt):].strip()
    
    print(resp)
    conversation.append(f"Assistant: {resp}")
