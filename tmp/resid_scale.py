"""Во сколько раз норма h больше единицы: подобрать разумный диапазон alpha.

SAE v5_32k обучен на layer-normed активациях, а стиринг мы делаем в сыром
резидуале. Значит alpha=1 в нормированном пространстве это alpha=||h|| в сыром.
Скрипт печатает ||h|| на слое 6 и типичную активацию латента.
"""

import argparse

import blobfile as bf
import sparse_autoencoder
import torch
import transformer_lens

LAYER = 6
HOOK = f"blocks.{LAYER}.hook_resid_post"
PROMPTS = [
    "The weather today is",
    "My favourite thing about this city is",
    "I spent the afternoon",
    "He opened the door and",
    "The Eiffel Tower stands in the middle of Paris",
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=int, default=27677)
    args = p.parse_args()

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device="cpu"
    )
    model.eval()

    norms = []
    caches = []
    for prompt in PROMPTS:
        _, cache = model.run_with_cache(model.to_tokens(prompt), remove_batch_dim=True)
        h = cache[HOOK]
        norms.append(h.norm(dim=-1))
        caches.append(h)
    norms = torch.cat(norms)
    print(f"||h|| на слое {LAYER}: медиана {norms.median():.1f}, "
          f"мин {norms.min():.1f}, макс {norms.max():.1f}")

    with bf.BlobFile(sparse_autoencoder.paths.v5_32k("resid_post_mlp", LAYER), mode="rb") as f:
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(torch.load(f))
    autoencoder.eval()

    acts = []
    for h in caches:
        latents, _ = autoencoder.encode(h)
        acts.append(latents[:, args.latent])
    acts = torch.cat(acts)
    print(f"активация латента {args.latent}: макс {acts.max():.3f}, среднее {acts.mean():.3f}")

    v = autoencoder.decoder.weight[:, args.latent]
    print(f"норма столбца декодера: {v.norm():.3f}")
    print(f"\nвклад латента в реконструкцию ~ act * ||v_dec|| = {acts.max() * v.norm():.2f}, "
          f"а ||h|| ~ {norms.median():.1f}")
    print("то есть чтобы направление что-то значило в сыром резидуале, alpha должна быть")
    print(f"порядка доли от ||h||: пробовать 0.5-4 x {norms.median():.0f} "
          f"= {0.5*norms.median():.0f}..{4*norms.median():.0f}")


if __name__ == "__main__":
    main()
