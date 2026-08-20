#!/usr/bin/env bash
# Полный протокол задания по порядку: сначала бейзлайн, потом методы.
#
#   1. проверить, что вектор делает то, что написано на этикетке
#   2. наивный стиринг h + alpha*v — точка отсчёта
#   3. обучить обе модели активаций на одних и тех же данных
#   4. сравнить три фронта на одних и тех же промптах и alpha
#
#   ./run_steering.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python -u"

$PY baseline.py --vector diffmean:sentiment --sweep --device mps > tmp/base_sentiment.log 2>&1
$PY train_denoiser.py --tag glp --objective flow --steps 15000 --device mps
$PY train_denoiser.py --tag mse --objective mse --steps 15000 --device mps
$PY eval_steering.py --tag pareto_sentiment --vector diffmean:sentiment \
  --repair none mse glp --mse runs/mse/denoiser.pt --glp runs/glp/denoiser.pt --device mps
