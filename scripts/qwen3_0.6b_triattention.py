from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from triattention.methods.triattention import apply_triattention_patch

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).to("cuda").eval()

apply_triattention_patch(
    model,
    stats_path=Path("triattention/calibration/custom/qwen3-0.6b.pt"),
    model_path=Path(model_id),
    kv_budget=128,
    divide_length=32,
    use_slack_trigger=True,
    count_prompt_tokens=True,
    allow_prefill_compression=True,
    per_head_pruning=True,
)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain how a transformer works in detail."}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=384,
        min_new_tokens=192,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(
    output[0, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
))

comp = model._triattention_compressor
print(f"\nAbsolute position: {comp.absolute_position}")
print(f"Retained cache entries per head: {len(comp.cache_positions)}")
print(f"Per-head compression occurred: {comp.cache_positions_per_head is not None}")