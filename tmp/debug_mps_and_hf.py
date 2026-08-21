"""Два вопроса, на которые проще ответить экспериментом, чем рассуждением.

1) NaN на MPS — это баг Apple-бэкенда в PyTorch или конкретно transformer_lens?
   Гоняем один и тот же GPT-2 через transformer_lens и через голый transformers.

2) Нужен ли вообще transformer_lens? Стиринг — это прибавить вектор к выходу слоя,
   в чистом transformers это делается обычным forward hook. Проверяем, что работает.

    python tmp/debug_mps_and_hf.py
"""

import torch

PROMPT = "The weather today is"
LAYER = 6


def check(name, logits):
    bad = (~torch.isfinite(logits)).sum().item()
    print(f"  {name:<28} finite={bad == 0}  плохих значений={bad}  "
          f"max={logits[torch.isfinite(logits)].max().item():.2f}")


def transformer_lens_forward(device):
    import transformer_lens

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=device
    )
    model.eval()
    with torch.no_grad():
        logits = model(model.to_tokens(PROMPT))
    check(f"transformer_lens [{device}]", logits)


def hf_forward(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    with torch.no_grad():
        logits = model(**tok(PROMPT, return_tensors="pt").to(device)).logits
    check(f"transformers [{device}]", logits)


def hf_steering(device, alpha=60.0):
    """Стиринг без transformer_lens: обычный forward hook на блоке трансформера."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    torch.manual_seed(0)
    v = torch.randn(model.config.n_embd, device=device)
    v = v / v.norm()

    def hook(module, args, output):
        # у GPT2Block выход это кортеж (hidden_states, ...)
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + alpha * v
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.transformer.h[LAYER].register_forward_hook(hook)
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=True,
                             top_k=50, pad_token_id=tok.eos_token_id)
    handle.remove()
    print(f"  стиринг через hook, alpha={alpha}: {tok.decode(out[0])!r}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, mps доступен: {torch.backends.mps.is_available()}")

    print("\nФорвард без всякого стиринга:")
    for device in ("cpu", "mps"):
        transformer_lens_forward(device)
        hf_forward(device)

    print("\nСтиринг без transformer_lens:")
    hf_steering("cpu")
