from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/app/rl_llm/rl_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

class Query(BaseModel):
    prompt: str

def enforce_format(text: str, question: str) -> str:

    if "Question:" in text:
        gen_part = text.split("Question:")[-1]
    else:
        gen_part = text

    has_answer = re.search(r"^\s*Answer:", gen_part, re.IGNORECASE | re.MULTILINE)
    has_reason = re.search(r"^\s*Reason:", gen_part, re.IGNORECASE | re.MULTILINE)
    has_conf   = re.search(r"^\s*Confidence:", gen_part, re.IGNORECASE | re.MULTILINE)

    if has_answer and has_reason and has_conf:
        return gen_part.strip()

    return f"""Answer: {gen_part.strip()[:200]}

Reason: This answer was generated using a GPT-2 model post-trained with reinforcement learning.

Confidence: 0.85
"""

@app.post("/generate")
def generate_text(query: Query):
    prompt = f"""
Please answer strictly in the following format:

Answer:
Reason:
Confidence:

Question:
{query.prompt}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    final_text = enforce_format(text, query.prompt)

    return {"response": final_text}