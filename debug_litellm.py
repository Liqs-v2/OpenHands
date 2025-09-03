import litellm
import pickle

api_key = 'sk-69Run1n1nd78J46W8A5unQ'
provider = 'litellm_proxy/gemini/gemini-2.5-flash'

with open("./agent_data.pkl", "rb") as f:
    agent_data = pickle.load(f)

print('EXAMPLE: Einstein Prompt\n')
print("Reasoning Effort: Disable")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    reasoning_effort='disable',
    messages=[
        {"role": "user", "content": "What is the cultural impact of Albert Einstein?"}
    ]
)

print(resp['model_extra']['usage']['completion_tokens_details'])
print()

print("Reasoning Effort: high")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    reasoning_effort='high',
    messages=[
        {"role": "user", "content": "What is the cultural impact of Albert Einstein?"}
    ]
)

print(resp['model_extra']['usage']['completion_tokens_details'])
print()

print("Thinking budget: 0")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    extra_body={
        "thinking": {"type": "enabled", "budget_tokens": 0}
    },
    messages=[
        {"role": "user", "content": "What is the cultural impact of Albert Einstein?"}
    ]
)

print(resp['model_extra']['usage']['completion_tokens_details'])
print("--------------------------------")

print("EXAMPLE: OpenHands django prompt\n")
print("Reasoning Effort: disable")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    reasoning_effort='disable',
    messages=agent_data['data']["messages"],
)

print(resp['model_extra']['usage']['completion_tokens_details'])
print()

print("Reasoning Effort: high")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    reasoning_effort='high',
    messages=agent_data['data']["messages"],
)

print(resp['model_extra']['usage']['completion_tokens_details'])
print()

print("Thinking budget: 0")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    extra_body={
        "thinking": {"type": "enabled", "budget_tokens": 0}
    },
    messages=agent_data['data']["messages"]
)
print(resp['model_extra']['usage']['completion_tokens_details'])
print()

print("Reasoning Effort: disable, User prompt only")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    reasoning_effort='disable',
    messages=agent_data['data']["messages"][1:2]
)
print(resp['model_extra']['usage']['completion_tokens_details'])
print()

print("Reasoning Effort: disable, OpenHands System prompt, Einstein User Prompt")
resp = litellm.completion(
    model=provider,
    api_key=api_key,
    base_url="https://litellm.labs.jb.gg/",
    reasoning_effort='disable',
    messages=[agent_data['data']["messages"][0], {"role": "user", "content": "What is the cultural impact of Albert Einstein?"}]
)
print(resp['model_extra']['usage']['completion_tokens_details'])
print()
