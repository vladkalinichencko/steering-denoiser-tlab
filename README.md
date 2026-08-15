# Steering GPT-2 without breaking it

Тестовое задание T-LAB, Mechanistic Interpretability. Условие — [NOTES.md](NOTES.md),
конвенции репозитория — [AGENTS.md](AGENTS.md).

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Baseline

Наивный стиринг `h <- h + alpha*v` после слоя 6, парето-кривая fluency/concept:

```bash
python baseline.py --latent 12345 --concept-words paris eiffel france --alphas 0 20 40 80 160
```

Результат — `runs/baseline_latent12345.json`.
