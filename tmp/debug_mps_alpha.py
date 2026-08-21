"""Проверка гипотезы: NaN на MPS появляется не сам по себе, а от большого alpha.

Голая генерация на MPS работает (см. debug_mps_generate.py). Значит подозрение на
переполнение: при большом alpha резидуал становится огромным, и MPS теряет то, что
CPU ещё вытягивает. Гоняем одинаковый стиринг на обоих устройствах.

    python tmp/debug_mps_alpha.py
"""

import torch
import transformer_lens

LAYER = 6
HOOK = f"blocks.{LAYER}.hook_resid_post"
ALPHAS = [0, 40, 80, 160, 320]


def run(device):
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=device
    )
    model.eval()
    torch.manual_seed(0)
    v = torch.randn(model.cfg.d_model, device=device)
    v = v / v.norm()

    print(f"\n[{device}]")
    for alpha in ALPHAS:
        hooks = [(HOOK, lambda resid, hook, a=alpha: resid + a * v)] if alpha else []
        torch.manual_seed(0)
        tokens = model.to_tokens(["The weather today is"] * 4)
        try:
            with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                logits = model(tokens)
                bad = (~torch.isfinite(logits)).sum().item()
                out = model.generate(tokens, max_new_tokens=24, do_sample=True,
                                     temperature=1.0, top_k=50, verbose=False)
            print(f"  alpha={alpha:>4}: форвард ок (NaN в логитах: {bad}), "
                  f"генерация ок  |h+av| max={logits.abs().max():.1f}")
        except Exception as exc:
            print(f"  alpha={alpha:>4}: ПАДАЕТ {type(exc).__name__}")


if __name__ == "__main__":
    for device in ("cpu", "mps"):
        run(device)
