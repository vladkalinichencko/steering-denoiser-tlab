"""Где именно на MPS появляется NaN: форвард в порядке, значит дело в генерации.

Проверяем четыре комбинации: transformer_lens / transformers x батч 1 / батч 4.
Батч важен, потому что в baseline.py генерится сразу несколько сэмплов на промпт,
и с батчем в игру вступает паддинг и KV-кэш.

    python tmp/debug_mps_generate.py
"""

import torch

PROMPT = "The weather today is"


def try_tl(device, batch):
    import transformer_lens

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=device
    )
    model.eval()
    torch.manual_seed(0)
    tokens = model.to_tokens([PROMPT] * batch)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=8, do_sample=True,
                             temperature=1.0, top_k=50, verbose=False)
    return model.to_string(out[0])


def try_hf(device, batch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    torch.manual_seed(0)
    inputs = tok([PROMPT] * batch, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=True,
                             top_k=50, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0])


if __name__ == "__main__":
    for name, fn in (("transformer_lens", try_tl), ("transformers", try_hf)):
        for device in ("cpu", "mps"):
            for batch in (1, 4):
                label = f"{name:<17} {device:<4} batch={batch}"
                try:
                    text = fn(device, batch)
                    print(f"OK    {label}  {text[:60]!r}", flush=True)
                except Exception as exc:
                    print(f"ПАДАЕТ {label}  {type(exc).__name__}: {str(exc)[:90]}",
                          flush=True)
