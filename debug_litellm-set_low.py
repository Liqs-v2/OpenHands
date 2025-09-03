import litellm
import pandas as pd

api_key = 'sk-'
provider = 'litellm_proxy/gemini/gemini-2.5-flash'
base_url = "https://litellm.labs.jb.gg/"

budgets = [0, 50, 100, 300, 600, 700, 725, 726, 727, 728, 800, 1200, 1500]
rows = []
# Use a prompt that is likely to require a lot of reasoning and tokens
max_reasoning_prompt = (
    "Please provide a detailed, step-by-step solution to the following problem, "
    "explaining every logical step and justification in depth. "
    "Problem: Prove that for every positive integer n, the sum of the cubes of the first n natural numbers "
    "is equal to the square of the sum of the first n natural numbers. "
    "That is, prove that 1^3 + 2^3 + ... + n^3 = (1 + 2 + ... + n)^2. "
    "Include all intermediate steps, explanations, and any relevant mathematical background."
)

for b in budgets:
    resp = litellm.completion(
        model=provider,
        api_key=api_key,
        base_url=base_url,
        extra_body={"thinking": {"type": "enabled", "budget_tokens": b}},
        # optional: try temperature to see if visible text changes while thinking stays stable
        # temperature=1.0, top_p=0.95,
        messages=[{"role": "user", "content": max_reasoning_prompt}]
    )
    dt = resp['model_extra']['usage']['completion_tokens_details']
    finish = resp['choices'][0].get('finish_reason')
    rows.append({"budget": b, "reasoning_tokens": dt.reasoning_tokens if dt else 0,
                 "finish_reason": finish})
df = pd.DataFrame(rows)
print(df)
print("--------------------------------")
