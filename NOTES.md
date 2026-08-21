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

## Зафиксированный протокол

| поле | решение |
|---|---|
| LM и точка вмешательства | GPT-2, `blocks.6.hook_resid_post`, \(d=768\) |
| данные денойзеров | FineWeb, context до 1024, все non-BOS позиции; 98/2 train/validation по hash документа; validation directions при обучении не используются |
| масштаб обучения | Mac smoke: первые 100k train activations; A100 screening/final: 100M activations, один проход; это reduced-scale проверка, а не буквальные 1B GLP |
| optimizer | AdamW, batch 4096 на A100, learning rate \(5\cdot10^{-5}\), cosine decay, warmup 1%; Mac batch выбирается по памяти без изменения effective batch |
| checkpoints | validation objective на фиксированном наборе каждые 1000 updates; сохраняются best и final; final methods повторяются с seeds 0/1/2 |
| основная direction | positive-sentiment DiffMean по SST-5 train: усреднить non-BOS states внутри каждого текста, затем взять разность class means; contrast data, prompts и generations не пересекаются |
| дополнительная проверка | OpenAI GPT-2 SAE directions только после отдельной проверки decoded effect; SAE encoder activation не считается property score |
| prompts | 100 neutral OpenWebText prefixes из DExperts, один и тот же зафиксированный список для всех методов |
| intervention scope | только новые response tokens; prompt, BOS и другие special tokens не меняются |
| generation | 20 новых токенов, temperature 1.0, top-k 50; Mac: seed 0 и 1 continuation/prompt; A100: seeds 0/1/2 и 5 continuations/prompt |
| steering strength | \(\alpha=r\,\overline{\lVert h\rVert_2}\), \(r\in\{0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0\}\) |
| property | mean positive probability локального SST-2 classifier по continuation; полный score distribution сохраняется |
| quality | conditional continuation NLL и perplexity под чистой GPT-2; dist-1/2/3 остаются дополнительной diversity-диагностикой |
| uncertainty | 95% bootstrap interval по prompt, 1000 resamples; одинаковые prompt/seed pairs между методами |
| порядок | validation direction и naive steering; MSE и GLP; ускорения; geometry; conditional methods; общий Pareto |
| локальный запуск | MPS, тот же код и настоящая архитектура, подмножество train documents, максимум 30 минут; результатом метода не считается |
| полный запуск | A100, полный train split; screening одним seed, финал тремя seeds |

## Общая архитектура денойзеров

Все unconditional denoisers получают одну стандартизованную активацию \(z\in\mathbb{R}^{768}\), поэтому attention не нужен. Simple MSE из задания использует один residual block `RMSNorm → Linear(768,3072) → GELU → Linear(3072,768)`. GLP и capacity-matched MSE используют `Linear(768,1536)`, четыре residual SwiGLU-блока с RMSNorm и hidden size 3072, затем `RMSNorm` и `Linear(1536,768)`. Additive MSE без времени получает только \(z\). Остальные flow/interpolation методы получают sinusoidal embeddings нужных времён через multiplicative modulation каждого SwiGLU gate.

## Эксперименты

| № | метод | что именно учим и делаем | код | статус | данные | результат | HTML |
|---|---|---|---|---|---|---|---|
| 1 | Validation direction | Проверить decoded property effect на полном alpha sweep до обучения repair; encoder и decoder стороны SAE проверяются раздельно | [baseline](baseline.py) | повторить | [SAE pilot](logs/check_sae27677.json), [latent 4875](logs/pareto_sae4875.json) | Два проверенных SAE decoder vectors не дали ожидаемого encoder effect | — |
| 2 | Noise tolerance | \(x=h+\sigma\epsilon\); найти, когда random noise меняет NLL, генерацию и downstream state | — | не начато | — | — | — |
| 3 | Naive | \(R(x)=x,\ x=h+\alpha v\); обязательная общая точка отсчёта | [baseline](baseline.py), [eval](eval_steering.py) | повторить | [старый pilot](logs/check_diffmeansentiment.json) | — | — |
| 4 | Additive MSE | Fixed-noise regression \(D(h+\sigma\epsilon)\to h\); один вызов, времени на входе нет; отдельно simple MLP и capacity-matched GLP body без timestep conditioning | [model](denoiser.py), [train](train_denoiser.py) | не запущен | — | — | — |
| 5 | Interpolation MSE | \(z_t=(1-t)h+t\epsilon\), сеть получает \(z_t,t\) и прямо предсказывает \(h\); один вызов | [model](denoiser.py), [train](train_denoiser.py) | текущий вариант повторить с \(t\)-conditioning | [старый train](logs/mse_history.jsonl) | — | — |
| 6 | GLP + SDEdit | Сеть предсказывает instantaneous velocity \(u_\theta(z_t,t)\approx\epsilon-h\); edited state зашумляется до \(t=0.5\), затем 20 Euler-шагов к \(t=0\) | [model](denoiser.py), [train](train_denoiser.py), [eval](eval_steering.py) | повторить | [старый train](logs/glp_history.jsonl), [старый Pareto](logs/pareto_sentiment.json) | — | — |
| 7 | One-Euler GLP | Один большой шаг старого GLP \(z_0\approx z_t-t\,u_\theta(z_t,t)\); без нового обучения, контроль ошибки интегратора | [model](denoiser.py), [eval](eval_steering.py) | код есть, не запущен | — | — | — |
| 8 | GLP endpoint distillation | Unconditional student \(S(z_t,t)\) повторяет endpoint 20-step GLP на random-corrupted natural activations; validation directions не видит | — | не начато | — | — | — |
| 9 | Consistency model | Endpoint network \(f_\theta(z_t,t)\to z_0\); точки одной teacher/EMA trajectory должны давать одинаковый endpoint, boundary \(f(z_0,0)=z_0\) задаётся skip connection | — | не начато | — | — | — |
| 10 | Rectified Flow | После teacher flow собрать пары noise/endpoints и переучить velocity на более прямые paths; сравнить 1/2/4 шага | — | не начато | — | — | — |
| 11 | MeanFlow | Сеть \(u_\theta(z_t,r,t)\) предсказывает average velocity на всём интервале \([r,t]\), то есть полный displacement за один шаг; objective требует JVP identity | — | не начато | — | — | — |
| 12 | Flow Map Matching | Сеть \(F_\theta(z_t,r,t)\) сразу предсказывает точку в другом времени; semigroup consistency сравнивает один большой переход с двумя малыми | — | не начато | — | — | — |
| 13 | Shortcut model | Сеть получает current time и step size \(d\); один большой displacement обучается совпадать с двумя рекурсивными half-steps, поэтому одна сеть поддерживает 1/2/4 шага | — | не начато | — | — | — |
| 14 | Nearest activation | Без обучения заменить \(x\) ближайшей real non-BOS activation; нижняя граница для методов, использующих activation bank | [repair](steering.py) | не начато | — | — | — |
| 15 | Segment-kNN | Среди real activations с проекцией вдоль отрезка \([h,h+\alpha v]\) выбрать точку с минимальной perpendicular distance | [repair](steering.py) | код есть, не запущен | — | — | — |
| 16 | Tangent/normal ablation | Для local SVD basis \(U(h)\): \(v_T=UU^Tv,\ v_N=(I-UU^T)v\); сравнить full, tangent и normal steering причинно | [split](steering.py), [eval](eval_steering.py) | повторить локально | [старый global-PCA pilot](logs/split_sentiment.json) | — | — |
| 17 | Tangent-preserving MSE | Сэмплировать tangent noise \(\tau=UU^T\epsilon_1\) и normal noise \(\nu=(I-UU^T)\epsilon_2\); учить \(D(h+\tau+\nu)\to h+\tau\) | [train](train_denoiser.py) | prototype есть, не запущен | — | — | — |
| 18 | Safe correction | Для correction \(c=D(x)-x\) использовать \(c_{safe}=c-\langle c,\hat v\rangle\hat v\); это control поверх любого repair, а не отдельный denoiser | [repair](steering.py) | код есть, не запущен | — | — | — |
| 19 | Local geodesic | Малые шаги \(h_{k+1}=h_k+\delta P_{T_{h_k}}v\); после каждого шага пересчитать kNN и local SVD basis | [prototype](steering.py) | код есть, не запущен | — | — | — |
| 20 | Curveball | Polynomial KPCA \(\phi(h)\), linear class-mean shift в kernel coordinates, kernel-weighted preimage и возврат residual; nonlinear, но inverse приближённый | — | не начато | — | — | — |
| 21 | Conditional field / UniSteer | Учить \(u_\theta(z_t,t,c)\) по activation-condition pairs; article baseline использует frozen text encoder и cross-attention, наш single-token вариант проверяет pooled condition + FiLM без attention | — | не начато | — | — | — |
| 22 | INNSteer | RealNVP affine coupling blocks \(\phi\) с ActNorm учат обратимые coordinates; в них делается linear mean-difference shift, затем точный \(\phi^{-1}\) даёт nonlinear input-dependent path | — | не начато | — | — | — |
| 23 | GLP prior during training | Учить direct steering function по property loss, GLP energy и minimal-movement penalty; на inference GLP не вызывается | — | не начато | — | — | — |
| 24 | GPT-2 MLP fine-tuning | Не основная линия: вопрос задания, можно ли переобучить штатный layer-6 MLP на denoising loss без adapter; запускать только после внешних denoisers | [model](denoiser.py), [train](train_denoiser.py) | отложено | — | — | — |
| 25 | Финальное сравнение | Все прошедшие baseline методы на одном протоколе, включая latency, Pareto, trajectories и failure cases | [HTML](viz.py) | не начато | — | — | — |

## Диагностика

![Единая геометрическая карта методов](assets/steering-geometry-concept.png)

Картинка выше объясняет координаты, но не является результатом. Каждый `runs/<experiment>/diagnostics.html` повторяет её по реальным активациям и содержит:

- clean \(h\), direction \(v\), steered \(x\), repaired \(\tilde h\) и correction \(\tilde h-x\);
- все denoising/flow states в одной held-out PCA-проекции, рядом с full-dimensional distances;
- kNN cloud, local tangent/normal basis, singular spectrum и effective rank;
- Jacobian singular values метода: amplification, contraction и стирание direction;
- \(\nabla_h L_{LM}\) и differentiable property gradient только для проверки их alignment с конкретным path;
- downstream layer-by-token drift, logits, decoded generations и causal alpha/repair ablations;
- train/validation objective, learning rate, gradient norm, checkpoints, latency и память.
