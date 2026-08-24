# Дешёвый steering GPT-2 без деградации текста

Тестовое задание T-LAB, направление mechanistic interpretability. Условие — в
NOTES.md, отчёт — в [REPORT.md](REPORT.md).

## Лучший чекпойнт

**https://huggingface.co/vladotpad/gpt2-one-euler-steering-denoiser**

Denoiser возвращает активацию, сдвинутую линейным steering, обратно к многообразию
естественных активаций за один шаг Эйлера вместо двадцати.

## Результаты

| метод | NLL | property | мс/токен |
|---|---:|---:|---:|
| наивный steering без починки | 3.073 | 0.785 | 0.275 |
| одношаговый MSE-denoiser | 3.279 | 0.795 | 0.251 |
| интерполяционный MSE | 3.099 | 0.833 | 1.739 |
| GLP, 20 шагов | 3.190 | 0.806 | 10.795 |
| GLP, один шаг Эйлера | 3.192 | 0.807 | 1.405 |
| MeanFlow | 3.190 | 0.799 | 1.721 |

Все методы ложатся практически на одну Pareto-линию, поэтому выигрыш здесь не в
качестве, а в стоимости: один шаг Эйлера даёт те же NLL и property в 7.7 раза быстрее
двадцати. Три training seeds, GPT-2 small, M1 Max.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python train_denoiser.py     # обучение denoiser на естественных активациях
python eval_steering.py      # оценка выбранных способов починки
python screening.py          # полный прогон всех методов -> screening.html
```

## Раскладка кода

| путь | что там |
|---|---|
| `steering.py` | направления, активации, хуки, генерация и метрики |
| `denoiser.py` | архитектура denoiser, flow-батчи и шаги возврата |
| `repairs/` | по файлу на способ починки: `none`, `mse`, `knn`, `glp`, `glp_one_euler` |
| `train_denoiser.py` | обучение denoiser |
| `eval_steering.py` | один общий evaluation path на все способы |
| `baseline.py` | наивный steering и sweep без починки |
| `screening.py`, `run_mac_final.py` | полный прогон и сборка страницы |
| `screening_template.html` | шаблон страницы, не отчёт |

Интерактивная диагностика: [screening.html](screening.html) — все точки, три
сида, доверительные интервалы, настоящие генерации и плашка пятнадцати более ранних
методов. Данные страницы лежат рядом в `runs/screening.json`.
