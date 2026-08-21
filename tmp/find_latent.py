"""Найти SAE-латент, отвечающий за концепт, контрастом двух наборов промптов.

Нужен, чтобы у baseline.py был не случайный вектор, а направление с понятным смыслом.
Скор латента = средняя активация на «позитивных» токенах минус на «негативных».

    python tmp/find_latent.py --topk 10

Выдаёт индексы латентов; дальше их проверяют глазами через SAE-viewer или прямо
стирингом в baseline.py.
"""

import argparse

import blobfile as bf
import sparse_autoencoder
import torch
import transformer_lens

LAYER = 6
LOCATION = "resid_post_mlp"
HOOK = f"blocks.{LAYER}.hook_resid_post"

POSITIVE = [
    "The Eiffel Tower stands in the middle of Paris",
    "We spent a week in Paris walking along the Seine",
    "Paris is the capital of France and its largest city",
    "French cuisine in Paris is famous around the world",
    "The Louvre in Paris holds the Mona Lisa",
    "A train from Lyon to Paris takes two hours",
]
NEGATIVE = [
    "The server stands in the middle of the rack",
    "We spent a week debugging the payment system",
    "Ottawa is the capital of Canada and a quiet city",
    "Regional cuisine here is famous around the world",
    "The museum downtown holds a modest collection",
    "A train from one town to the next takes two hours",
]


@torch.no_grad()
def mean_latents(model, autoencoder, prompts):
    """Средняя активация каждого латента по всем токенам набора."""
    total = None
    count = 0
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        latents, _ = autoencoder.encode(cache[HOOK])  # (n_tokens, n_latents)
        total = latents.sum(0) if total is None else total + latents.sum(0)
        count += latents.shape[0]
    return total / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=args.device
    )
    model.eval()

    with bf.BlobFile(sparse_autoencoder.paths.v5_32k(LOCATION, LAYER), mode="rb") as f:
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(torch.load(f))
    autoencoder.to(args.device).eval()

    pos = mean_latents(model, autoencoder, POSITIVE)
    neg = mean_latents(model, autoencoder, NEGATIVE)
    score = pos - neg

    print(f"{'latent':>8} {'pos':>10} {'neg':>10} {'diff':>10}")
    for idx in torch.topk(score, args.topk).indices.tolist():
        print(f"{idx:>8} {pos[idx]:>10.3f} {neg[idx]:>10.3f} {score[idx]:>10.3f}")


if __name__ == "__main__":
    main()
