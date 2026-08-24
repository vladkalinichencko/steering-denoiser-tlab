# Стиринг GPT-2 без деградации текста

Денойзер возвращает активацию, сдвинутую линейным стирингом, обратно к естественным активациям.

Отчёт — [report.md](report.md). Диагностика — [screening.html](screening.html).
Лучший чекпойнт — https://huggingface.co/vladotpad/gpt2-one-euler-steering-denoiser

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_denoiser.py    # обучение денойзера на естественных активациях
python eval_steering.py     # оценка выбранных способов починки
python screening.py         # полный прогон всех методов, пересборка screening.html
```

## Где что лежит

| из отчёта | в коде |
|---|---|
| наивный стиринг без починки | `baseline.py`, `repairs/none.py` |
| аддитивный денойзер | `repairs/mse.py` |
| GLP на двадцать шагов | `repairs/glp.py` |
| один шаг Эйлера | `repairs/glp_one_euler.py` |
| ближайшая активация на отрезке | `repairs/knn.py` |
| архитектура денойзера и шаги возврата | `denoiser.py` |
| направления, активации, хуки, метрики | `steering.py` |
| обучение денойзера | `train_denoiser.py` |
| общий путь оценки | `eval_steering.py` |
| полный прогон и сборка страницы | `screening.py`, `run_mac_final.py` |
| шаблон страницы | `screening_template.html` |
