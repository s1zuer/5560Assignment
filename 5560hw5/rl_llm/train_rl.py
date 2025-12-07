import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward import compute_reward

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

prompts = [
    "What is machine learning?",
    "What is reinforcement learning?",
    "Explain overfitting in simple terms."
]

EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")
    total_reward = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True
        )

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        reward = compute_reward(gen_text)
        total_reward += reward

        logits = model(**inputs).logits
        log_probs = F.log_softmax(logits, dim=-1)

        last_log_prob = log_probs[0, -1].mean()

        loss = -last_log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Prompt:", prompt)
        print("Generated:", gen_text)
        print("Reward:", reward)

    print("Average Reward:", total_reward / len(prompts))

model.save_pretrained("./rl_model")
tokenizer.save_pretrained("./rl_model")

print("\n RL Post-training finished. Model saved to ./rl_model")