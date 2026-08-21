# Mechanistic Interpretability — дешёвый стиринг без деградации

## Задание

### О чём это

Интерпретируемость даёт возможность управлять вычислениями в языковых моделях.
Самый многообещающий способ на данный момент — **steering**. Пусть найдено
направление $v \in \mathbb{R}^d$ (например, столбец декодера SAE), отвечающее за
свойство модели, которое мы хотим усилить. Тогда мы делаем интервенцию

$$\tilde{h} = h + \alpha v,$$

где $h \in \mathbb{R}^d$ — скрытое состояние модели, $\alpha \in \mathbb{R}_{+}$ —
сила применения вектора. Обычно для появления эффекта нужны достаточно большие
$\alpha$, превосходящие норму исходного $h$. Из-за этого модель часто ломается
(растёт перплексия).

Демо: <https://huggingface.co/spaces/dlouapre/eiffel-tower-llama>

### Задача

Изучить, как уменьшить негативный эффект стиринга.

**Как измерять негативный эффект.** Две оси: по $x$ — что-то связанное с качеством
текста (fluency score, perplexity), по $y$ — насколько нужное свойство присутствует
в сгенерированных текстах. Изменяя $\alpha$, получаем парето-фронт в этих осях.
Чем сильнее свойство, тем хуже связность; хотим, чтобы кривая лежала как можно
ближе к правому верхнему углу.

**Как можно улучшить.** В недавней работе GLP [1] предлагается обучить flow
matching для активаций, чтобы затем аккуратно «расшумлять» их после стиринга.
Однако это долго обучать и очень дорого для инференса. Предлагается обучить
простую модель

$$\hat{h} = \operatorname{denoiser}(h + \varepsilon), \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I),$$

$$\mathcal{L} = \lVert h - \hat{h} \rVert_2^2,$$

после чего во время инференса делать

$$\tilde{h} = \operatorname{denoiser}(h + \alpha v).$$

Для простоты фиксируем, что интервенции проводятся после срединного слоя LM
(например, после слоя 6 для GPT-2, у которой 12 слоёв).

Обязательно подумать про детали реализации:

- Как лучше зашумлять $h$? Помимо $h + \varepsilon$ можно подумать о
  $t \cdot h + (1 - t) \cdot \varepsilon$, $t \sim \mathcal{U}[0; 1]$.
- Какая архитектура должна быть у денойзера.
- Как можно исправлять ошибки стиринга — возможно, есть что-то хитрее, чем просто
  $\operatorname{denoiser}(h + \alpha v)$.
- Можно ли просто дообучить на такой лосс уже существующие в модели MLP, чтобы не
  менять структуру модели.
- Любая другая идея, как улучшить стиринг для LM.

Выше приведён лишь пример того, что можно сделать. Если есть другие идеи или они
придут во время выполнения задания — обязательно отразить это в отчёте и
попробовать свой метод.

### План действий

1. Выбрать небольшую LM и векторы $v$, на которых будет валидация (в минимальном
   варианте — GPT-2 и SAE к ней [2]).
2. Придумать, как измерять fluency/concept score. В идеале взять пайплайн из
   persona vectors, но он требует API до ChatGPT. Обратить внимание на
   dist-1, dist-2, dist-3 — их посчитать проще всего, и они что-то говорят о fluency.
3. Провалидировать векторы в простом сетапе стиринга $\tilde{h} = h + \alpha v$;
   убедиться, что получается похожая картинка с примером.
4. Обучить свою модель. **Важно: она не должна знать про существование каких-то $v$
   из валидации.**
5. Только после этого сравнивать обученную модель с обычным стирингом.
6. Написать отчёт о проделанной работе.

### Оформление решения

Код и отчёт (TeX или markdown) — в репозитории на GitHub, лучший адаптер или
чекпойнт — в открытый репозиторий на Hugging Face.

### Критерии

Важно не просто закинуть в GPT эту страницу и посмотреть на отчёт в конце: сильно
поощряются идеи интереснее описанных выше и демонстрация того, что они работают
лучше. Кроме того, не стоит забывать про анализ метода: метод, который хорошо
работает, — это лишь половина статьи, вторая половина — показать, **почему** и
**как** он работает.

### Полезные репозитории и ссылки

- <https://github.com/TransformerLensOrg/TransformerLens>
- <https://github.com/decoderesearch/SAELens>
- <https://github.com/safety-research/persona_vectors>

### Ссылки

[1] Luo, Feng, Darrell, Radford, Steinhardt. *Learning a Generative Meta-Model of
LLM Activations* (GLP). <https://generative-latent-prior.github.io/>
[2] <https://github.com/openai/sparse_autoencoder>

---

## Текущее состояние

На 2026-08-21 в репозитории нет валидного сравнения методов. GLP- и MSE-чекпойнты
подтверждают, что существующий training path исполнялся и loss снижался, но их данные
загрязнены BOS-активациями. Сохранённый Pareto не интерпретируется: conditional
perplexity вычислена с повторным BOS и включает предсказание последнего токена
prompt. Экспериментальный код не менялся до согласования владельца.

## Идеи и их происхождение

- **Задание:** наивный steering \(h+\alpha v\); MSE-денойзер для аддитивного
  гауссовского шума; возможная interpolation corruption; выбор архитектуры; варианты
  ремонта; fine-tuning существующего LM MLP; GPT-2 + OpenAI SAE как минимальный
  setup; property/quality Pareto; dist-1/2/3 как возможная упрощённая fluency
  диагностика; Persona Vectors как предпочтительный evaluation pipeline.
- **GLP:** flow matching на однотокенных стандартизованных активациях, SDEdit после
  steering и paper-specific sentiment evaluation. Источник:
  <https://arxiv.org/html/2602.06964>.
- **OpenAI SAE:** `v5_32k` действительно предназначен для GPT-2 layer 6
  `resid_post_mlp`, использует TopK-32 и нормализует вход в `encode`. Столбец decoder
  разрешён заданием как направление steering; top activations и direct logit lens —
  только прокси для интерпретации, а не доказательство downstream-эффекта. Источник:
  `ext/sparse_autoencoder/README.md`, `paths.py`, `model.py`.
- **Persona Vectors:** contrastive prompts/questions, response-average-difference
  directions, response-only steering и LLM judge. В текущем коде этот pipeline не
  реализован. Источник: <https://github.com/safety-research/persona_vectors>.
- **Владелец:** диагностика должна показывать реальные активации, direction,
  intervention, denoiser input/output/residual, decoded generations, failure cases и
  causal ablations; градиенты нужны только под конкретную гипотезу.
- **Claude, не задание и не проверенная реализация статьи:** latent `27677`, наборы
  positive/negative texts, шесть prompts, абсолютные alpha, SST-2 score, четыре блока
  денойзера, 500k активаций, 15k steps, PCA/rank/nearest-neighbour diagnostics,
  `geodesic_hook`, directional/relative/gated repair и one-step Euler. Эти варианты
  нельзя выдавать за требование или результат метода без отдельного согласования.
- **Литература, упомянутая в старых заметках, но не реализованная:** Curveball
  Steering — polynomial kernel PCA/preimage, arXiv:2603.09313; UniSteer — conditional
  activation-space flow matching, arXiv:2605.30076; INNSteer — arXiv:2606.08454.
  Ссылки 2405.17130, 1711.01573 и 2206.00934 не описывают приписанные им steering-
  методы, поэтому прежние выводы из них сняты.

## Доказанные факты аудита

### Данные и checkpoints

- `datasets/activations_gpt2_layer6.pt` имеет shape `[500000, 768]`. В нём 4015
  векторов с нормой больше 500; они стоят через границы документов и имеют норму
  около 3087. `collect_activations()` вызывает `model.to_tokens()` с default BOS и
  сохраняет все позиции, поэтому это BOS-активации. GLP, напротив, исключает BOS.
- Оба checkpoint обучены на одном загрязнённом cache. GLP содержит 82,600,704
  trainable parameters, MSE — 58,992,384; равенства архитектур нет, поэтому текущий
  Pareto одновременно меняет objective и capacity.
- Оба запуска дошли до step 14999. Эффективные сохранённые настройки: 500k vectors,
  batch 1024, 4 blocks, width multiplier 2, expansion 2, interpolation corruption,
  seed 0, MPS, AdamW с `lr=3e-4`, cosine schedule без warmup. GLP best validation
  loss — 0.891665, MSE — 0.312304. Эти числа проверяют только соответствующие
  objectives на текущем cache и не являются quality/property результатом.
- Config/checkpoint не фиксируют git commit, точную команду, package versions,
  dataset revision/document IDs или hash cache. Checkpoint не содержит состояния
  optimizer/scheduler, поэтому provenance и возобновление неполны.
- Validation — первые 2% последовательного cache, а не предварительно выбранный
  random/document split. Fixed seed есть, но воспроизводимость cache не зафиксирована.

### Evaluation

- `generate()` декодирует последовательность вместе с BOS. `perplexity()` повторно
  токенизирует её с default BOS, добавляет второй BOS и начинает slice на
  `n_prompt - 1`; метрика включает последний prompt transition. Все сохранённые PPL
  не являются заявленной conditional continuation perplexity.
- Steering и repair применяются ко всем token positions, включая BOS. Область
  вмешательства не была согласована и расходится с response-only режимом Persona
  Vectors.
- SAE latent `27677` выбран после просмотра contrast prompts, а затем проверялся на
  близком концепте. Это selection leakage; вектор не был зафиксирован как validation
  direction до обучения денойзеров.
- MPS-check на двух текстах для обычного forward steering дал разницу CPU/MPS loss
  не больше \(3\cdot10^{-5}\) при \(\alpha\in\{0,20,80,160\}\). Он не проверяет
  MPS-training, SDEdit, generation или детерминизм.

### Соответствие первоисточникам

- Текущий flow objective и Euler integration соответствуют базовым формулам GLP:
  \(z_t=(1-t)z_0+t\varepsilon\), target \(\varepsilon-z_0\). Gated residual MLP и
  multiplicative timestep modulation также соответствуют механизму статьи.
- Численный протокол не соответствует GLP. Статья использует 1B non-BOS FineWeb
  activations, one epoch, batch 4096, `lr=5e-5`, cosine schedule с warmup ratio 0.01,
  а для меньшей Llama — depth 3/6/12/24. Текущие 500k, примерно 30 epochs,
  batch 1024, `lr=3e-4`, no warmup и depth 4 — несогласованная Mac-адаптация.
- GLP SDEdit использует standardized edit, \(t_{start}=0.5\), 20 reverse steps и
  alpha как множитель средней нормы активации. Текущий sweep использует абсолютные
  alpha `[0, 10, 20, 40, 80, 160]`.
- GLP sentiment evaluation использует DiffMean, 100 neutral OpenWebText prefixes,
  20 generated tokens и относительные coefficients; quality/property оцениваются
  либо LLM judge, либо в дополнительном эксперименте conditional NLL и 5-point
  SetFit sentiment с 1000 prefixes и 95% bootstrap intervals. Текущие восемь
  contrast sentences, шесть prompts, 30 tokens, binary SST-2, один seed и отсутствие
  intervals — отдельный протокол Claude.
- `geodesic_hook` не является Curveball Steering: Curveball использует polynomial
  kernel PCA, preimage и residual preservation. One-step GLP Euler не является
  MeanFlow или consistency distillation.

### Диагностика и артефакты

- `runs/report.html` содержит Pareto/training curves и по одному sample на точку, но
  не реальные activation tensors, direction, intervention, denoiser input/output,
  residual, gradients, causal ablations или полный набор decoded generations.
- `tmp/diag_steering.py` и `tmp/diag_denoise_path.py` используют старый класс и старую
  checkpoint schema, поэтому несовместимы с текущим кодом. Их PNG/JSON — исторические
  артефакты, а не диагностика текущих checkpoints.
- Старые norm z-score, nearest-neighbour distance, фильтр `>5× median`, PCA, logit
  lens и propagation plots не были выведены из заранее записанной гипотезы. Они не
  используются как доказательство метода.
- `Training Observer`, Anyflow и ClearML в проекте не найдены. Их нельзя считать
  частью текущего runtime до согласованной интеграции.

## Единый реестр экспериментов

| Эксперимент | Основание | Код | Статус | Данные/запуск | Результат | Диагностика/логи |
|---|---|---|---|---|---|---|
| Проверка SAE latent 27677 и наивный alpha sweep | Идея Claude; SAE direction разрешён заданием | `baseline.py`, `steering.py` | Некорректный выбор validation direction | Latent выбран после contrast search; related prompts использованы повторно | Не доказывает валидность direction | `runs/baseline_latent27677.json`, `runs/sweep_latent27677.json` |
| Обучение GLP flow adaptation | GLP | `denoiser.py`, `train_denoiser.py` | Проверка проводки; методологически загрязнено | `runs/glp/config.json`, checkpoint step 14999; BOS cache; несогласованная reduced-scale config | Best val flow loss 0.891665; method claim отсутствует | `runs/glp/history.jsonl`, `runs/glp/denoiser.pt` |
| Обучение MSE interpolation denoiser | Задание предлагает MSE и interpolation corruption как варианты | `denoiser.py`, `train_denoiser.py` | Проверка проводки; методологически загрязнено | `runs/mse/config.json`, checkpoint step 14999; BOS cache; несогласованная architecture | Best val MSE 0.312304; method claim отсутствует | `runs/mse/history.jsonl`, `runs/mse/denoiser.pt` |
| Сравнение none / MSE / GLP на sentiment | Протокол Claude | `eval_steering.py`, `steering.py`, `viz.py` | Невалидно | 6 prompts, 48 samples/alpha, absolute alpha; неверная PPL; capacity mismatch | Pareto не интерпретируется | `runs/pareto_sentiment.json`, `runs/report.html` |
| Проверка CPU/MPS обычного forward steering | Разовая проверка проводки | `tmp/check_mps.py` | Завершено в текущей среде | 2 texts, alpha 0/20/80/160 | Max loss difference \(3\cdot10^{-5}\); вывод ограничен forward loss | Терминальный вывод 2026-08-21; отдельный artifact не сохранён |
| Валидный наивный steering baseline | Задание; точный evaluation protocol выбирает владелец |  | Не начато |  |  |  |
| Валидная reduced-scale GLP baseline | GLP; численные отклонения согласует владелец |  | Не начато |  |  |  |
| Валидный MSE denoiser | Задание; corruption и architecture согласует владелец |  | Не начато |  |  |  |
| Fine-tuning существующего LM MLP на denoising loss | Вопрос задания |  | Не начато |  |  |  |
| Curveball Steering baseline | Curveball Steering, arXiv:2603.09313 |  | Не начато; включение не согласовано |  |  |  |
| UniSteer baseline | UniSteer, arXiv:2605.30076 |  | Не начато; включение не согласовано |  |  |  |
| INNSteer baseline | INNSteer, arXiv:2606.08454 |  | Не начато; включение не согласовано |  |  |  |

## Предлагаемая архитектура, не согласована

Минимальный вариант сохраняет существующие `steering.py` и `denoiser.py` как обычные
модули. Фиксированный Python entry script из `tmp/` передаёт одну конфигурацию в
общие функции сбора данных, training и evaluation; evaluation пишет один JSON, а
`viz.py` строит из него один самодостаточный HTML. Отдельные методы меняют только
согласованную функцию repair/objective, а prompts, directions, token scope,
generation и метрики остаются общими.

До переписывания нужно отдельно решить, сохраняем ли текущие root entry scripts как
общий path или переносим экспериментальную обвязку в `tmp/`. Автоматический skip по
наличию checkpoint нужно убрать: завершённость и provenance должны проверяться по
содержимому артефакта.

## Предлагаемый диагностический контракт, не согласован

Для каждой точки \(\alpha\) HTML должен связать один raw JSON с prompt, токенами,
активацией \(h\), direction \(v\), \(h+\alpha v\), выходом denoiser, repair residual,
decoded continuation и обеими целевыми метриками. Causal panel сравнивает на тех же
seed `alpha=0`, naive steering и каждый согласованный repair. Failure cases выбираются
по заранее заданному правилу из целевых метрик, а не вручную после просмотра.

Градиенты добавляются только после выбора differentiable hypothesis. Например,
можно проверить, уменьшает ли repair компонент изменения, направленный по градиенту
quality loss, сохраняя компонент по согласованному property gradient. Этот тест имеет
смысл только если владелец утвердит loss и точку градиента; без этого gradients не
считаются.

## Вопросы владельцу перед кодом

1. Какой один основной evaluation protocol фиксируем: адаптацию GLP sentiment
   protocol к GPT-2; Persona Vectors с LLM judge и другой подходящей LM; или отдельный
   GPT-2/SAE protocol с заранее выбранными SAE directions и согласованным property
   scorer? Первый вариант требует меньше всего новых решений, но не проверяет SAE
   directions из минимального плана задания.
2. Где применять steering: ко всем non-BOS positions, только к continuation после
   prompt или иначе? Это меняет и causal interpretation, и сопоставимость с Persona
   Vectors.
3. Какую архитектуру сравнивать с MSE: одинаковый с GLP gated residual MLP для
   capacity control; более простой MLP как буквальная «простая модель» задания; или
   оба варианта отдельными строками? Текущие GLP и MSE имеют разное число параметров.
4. Какие reduced-scale параметры разрешены для Mac-проверки и какие paper-scale
   параметры обязательны на A100? Отдельно нужно утвердить corpus size, split,
   batch, steps/epochs, depth, learning rate, warmup, seeds и stopping rule.
5. Какие validation directions и contrast/prompt sets фиксируем до нового обучения?
   Latent `27677` нельзя использовать как независимую validation direction без новой
   predeclared проверки либо отдельного held-out набора.
6. Нужен ли в первом сравнении только обязательный набор `naive / GLP / MSE`, или
   добавлять Curveball, UniSteer, INNSteer и fine-tuned LM MLP? Это разные объёмы
   работы и разные утверждения, поэтому их нельзя молча включить в один этап.
7. Утверждаем ли предложенный HTML-контракт и gradient hypothesis выше? Без ответа
   фиксируются только реальные tensors/generations и causal alpha ablations, без PCA,
   rank, nearest-neighbour heuristics и новых proxy-метрик.
