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
| масштаб обучения | Mac: до 100k train activations, короткая проверка динамики и диагностики; текущий full-body preliminary использует 100k train и 2k validation. A100: 100M activations, один проход; это reduced-scale проверка, а не буквальные 1B GLP |
| optimizer | AdamW, batch 4096 на A100, learning rate \(5\cdot10^{-5}\), cosine decay, warmup 1%; Mac batch выбирается по памяти без изменения effective batch |
| checkpoints | validation objective на фиксированном наборе каждые 1000 updates; сохраняются best и final; final methods повторяются с seeds 0/1/2 |
| основная direction | positive-sentiment DiffMean по SST-5 train: усреднить non-BOS states внутри каждого текста, затем взять разность class means; contrast data, prompts и generations не пересекаются |
| дополнительная проверка | OpenAI GPT-2 SAE directions только после отдельной проверки decoded effect; SAE encoder activation не считается property score |
| prompts | 100 neutral OpenWebText prefixes из DExperts, один и тот же зафиксированный список для всех методов |
| intervention scope | только новые response tokens; prompt, BOS и другие special tokens не меняются |
| generation | 20 новых токенов, temperature 1.0, top-k 50; Mac: seed 0 и 1 continuation/prompt; A100: seeds 0/1/2 и 5 continuations/prompt |
| steering strength | \(\alpha=r\,\overline{\lVert h\rVert_2}\), \(r\in\{0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0\}\) |
| generative repair | GLP, Consistency и MeanFlow: Mac sweep \(t_{start}\in\{0.2,0.35,0.5\}\); Rectified Flow: \(t_{start}=0.2\) и 1/2/4 шага. Выбор по validation prompts фиксируется до A100 evaluation |
| property | mean positive probability локального SST-2 classifier по continuation; полный score distribution сохраняется |
| quality | conditional continuation NLL и perplexity под чистой GPT-2; dist-1/2/3 остаются дополнительной diversity-диагностикой |
| uncertainty | 95% bootstrap interval по prompt, 1000 resamples; одинаковые prompt/seed pairs между методами |
| порядок | validation direction и naive steering; MSE и GLP; ускорения; geometry; conditional methods; общий Pareto |
| локальный запуск | MPS, тот же код и настоящая архитектура, подмножество train documents, максимум 30 минут; сравниваем сходимость, устойчивость и реальные траектории, но не финальное качество метода |
| полный запуск | A100, полный train split; screening одним seed, финал тремя seeds |
| общий Mac-report | один зафиксированный PCA basis, fit на clean FineWeb validation activations; все методы, alpha и trajectories преобразуются только через него |
| geometry screen | 20k natural non-BOS activations; для каждой точки 256 соседей и local SVD rank 16; показываются весь singular spectrum, kNN distance, local-PCA residual, tangent/normal displacement и correction energy capacity-MSE |
| noise tolerance | isotropic Gaussian масштабируется так, чтобы ожидаемая норма perturbation была (r\,\overline{\lVert h\rVert_2}); используется тот же (r\)-grid и seed 0 |

## Общая архитектура денойзеров

Все unconditional denoisers получают одну стандартизованную активацию \(z\in\mathbb{R}^{768}\), поэтому attention не нужен. Simple MSE из задания использует один residual block `RMSNorm → Linear(768,3072) → GELU → Linear(3072,768)`. GLP и capacity-matched MSE используют `Linear(768,1536)`, четыре residual SwiGLU-блока с RMSNorm и hidden size 3072, затем `RMSNorm` и `Linear(1536,768)`. Additive MSE без времени получает только \(z\). Flow/interpolation методы получают sinusoidal embeddings нужных времён через multiplicative modulation каждого SwiGLU gate.

Первый Mac wiring-run уменьшает общий GLP-backbone до width 768, двух SwiGLU-блоков и hidden size 1536. Full-body Mac preliminary и A100 используют width 1536, четыре блока и hidden size 3072. Additive capacity control меняет только timestep path, поэтому совпадает с GLP по основным блокам, но не по общему числу параметров conditioning.

Для conditional field cross-attention избыточен: условие одно и не образует последовательность. Frozen text encoder даёт pooled vector \(e(c)\); MLP из \(e(c)\) и timestep embedding предсказывает \(\gamma,\beta\), а каждый SwiGLU-блок применяет FiLM к gate:

$$g'=(1+\gamma)\odot g+\beta.$$

## Конструктор методов

| деталь | варианты |
|---|---|
| что предсказывает сеть | clean endpoint; instantaneous velocity; average velocity |
| что получает сеть | activation; time; interval; pooled concept condition |
| где происходит edit | исходное пространство; local tangent space; kernel coordinates; invertible coordinates |
| сколько вызовов | 1; 2–4; 20 |

## Эксперименты

| № | метод | что именно учим и делаем | код | статус | данные | предварительный Mac | итоговый A100 | HTML |
|---|---|---|---|---|---|---|---|---|
| 1 | Validation direction | Проверить decoded property effect на полном alpha sweep до обучения repair; encoder и decoder стороны SAE проверяются раздельно | [baseline](baseline.py) | повторить | [SAE pilot](logs/check_sae27677.json), [latent 4875](logs/pareto_sae4875.json) | Два SAE decoder vectors не дали ожидаемого encoder effect | — | — |
| 2 | Noise tolerance | \(x=h+\sigma\epsilon\); найти, когда random noise меняет NLL, generation и downstream state | — | не начато | — | — | — | — |
| 3 | Naive | \(R(x)=x,\ x=h+\alpha v\); общая точка отсчёта | [diagnostics](tmp/diagnostics.py) | предварительный Mac | 3 diagnostic prompts, seed 0 | \(r=0.8\): NLL 4.539, positive 0.999; \(r=1.6\): NLL 5.577, positive 1.000 | — | [HTML](runs/mac_full_additive_simple/diagnostics.html) |
| 4 | Additive MSE | Fixed-noise regression \(D(h+\sigma\epsilon)\to h\); отдельно simple MLP и capacity-matched GLP body, оба без timestep | [model](tmp/methods.py), [train](tmp/training.py), [run](tmp/run_mac_baselines.py) | предварительный Mac | FineWeb 100k/2k activations, 2000 updates, MPS | simple val 1.039→0.636; capacity val 1.333→0.487. Capacity при \(r=1.6\): NLL 5.577→5.084, positive 1.000→1.000 на 3 prompts | — | [simple](runs/mac_full_additive_simple/diagnostics.html), [capacity](runs/mac_full_additive_capacity/diagnostics.html) |
| 5 | Interpolation MSE | \(z_t=(1-t)h+t\epsilon\); сеть получает \(z_t,t\) и предсказывает \(h\) за один вызов | [model](tmp/methods.py), [train](tmp/training.py) | предварительный Mac | FineWeb 100k/2k activations, 2000 updates, MPS | val 1.018→0.605; при \(r=0.8\): NLL 4.539→4.324, positive 0.999→1.000 на 3 prompts | — | [HTML](runs/mac_full_interpolation/diagnostics.html) |
| 6 | [GLP](https://arxiv.org/abs/2602.06964) + SDEdit | \(u_\theta(z_t,t)\approx\epsilon-h\); steered state частично зашумляется, затем 20 Euler-шагов к \(t=0\) | [model](tmp/methods.py), [train](tmp/training.py) | предварительный Mac | FineWeb 100k/2k activations, 2000 updates, MPS | val 2.321→1.712; sweep выбрал \(t_{start}=0.2\): при \(r=0.8\) NLL 4.539→4.078, positive 0.999→0.999 на 3 prompts | — | [HTML](runs/mac_full_glp/diagnostics.html) |
| 7 | One-Euler GLP | Один большой шаг \(z_0\approx z_t-t\,u_\theta(z_t,t)\); без нового обучения, контроль ошибки интегратора | [model](tmp/methods.py) | не начато | — | — | — | — |
| 8 | [Consistency model](https://arxiv.org/abs/2303.01469) | Endpoint network \(f_\theta(z_t,t)\to z_0\); точки одной EMA trajectory дают одинаковый endpoint, \(f(z_0,0)=z_0\) задаётся skip connection | [model](tmp/methods.py), [run](tmp/run_mac_speedups.py) | предварительный Mac; не прошёл текущий экран | FineWeb 20k/1k activations, 300 updates, CPU | val 0.00716→0.00606; лучший repair при \(r=0.8\): NLL 4.539→4.358, но positive 0.999→0.637 на 3 prompts | — | [HTML](runs/mac_reduced_consistency/diagnostics.html) |
| 9 | [Rectified Flow](https://arxiv.org/abs/2209.03003) | 20-step GLP создаёт noise/endpoint pairs; новое velocity field выпрямляет эти paths; inference сравнивается при 1/2/4 шагах | [model](tmp/methods.py), [pairs/train](tmp/training.py), [run](tmp/run_mac_speedups.py) | предварительный Mac | FineWeb 20k/1k activations; 512/128 teacher pairs, seed 0; 300 updates, CPU | val 0.287→0.130; при \(r=0.8\) один шаг даёт NLL 4.539→4.278, positive 0.999→0.979; 2–4 шага сильнее снижают property на 3 prompts | — | [HTML](runs/mac_reduced_rectified/diagnostics.html) |
| 10 | [MeanFlow](https://arxiv.org/abs/2505.13447) | \(u_\theta(z_t,r,t)\) предсказывает average velocity на \([r,t]\), то есть displacement за один шаг; objective использует JVP \((v,0,1)\) и stop-gradient target | [model](tmp/methods.py), [run](tmp/run_mac_speedups.py) | предварительный Mac | FineWeb 20k/1k activations, 300 updates, CPU | unweighted val MSE 1806.9→1693.4; \(t_{start}=0.2\): при \(r=0.8\) NLL 4.539→4.231, positive 0.999→0.999; при \(r=1.6\) NLL 5.577→5.193, positive 1.000→0.999 на 3 prompts | — | [HTML](runs/mac_reduced_meanflow/diagnostics.html) |
| 11 | Nearest activation | Без обучения заменить \(x\) ближайшей real non-BOS activation | [repair](steering.py) | не начато | — | — | — | — |
| 12 | Segment-kNN | Среди real activations с проекцией на \([h,h+\alpha v]\) выбрать точку с минимальной perpendicular distance | [repair](steering.py) | не начато | — | — | — | — |
| 13 | Tangent/normal ablation | Для local SVD basis \(U(h)\): \(v_T=UU^Tv,\ v_N=(I-UU^T)v\); сравнить full, tangent и normal steering | [split](steering.py), [eval](eval_steering.py) | повторить локально | [старый global-PCA pilot](logs/split_sentiment.json) | — | — | — |
| 14 | Tangent-preserving MSE | \(\tau=UU^T\epsilon_1,\ \nu=(I-UU^T)\epsilon_2\); учить \(D(h+\tau+\nu)\to h+\tau\) | [train](train_denoiser.py) | не начато | — | — | — | — |
| 15 | Safe correction | Для \(c=D(x)-x\) использовать \(c_{safe}=c-\langle c,\hat v\rangle\hat v\), чтобы repair не стирал steering coordinate | [repair](steering.py) | не начато | — | — | — | — |
| 16 | Local geodesic | \(h_{k+1}=h_k+\delta P_{T_{h_k}}v\); после шага пересчитать kNN и local SVD basis | [prototype](steering.py) | не начато | — | — | — | — |
| 17 | Curveball | Polynomial KPCA \(\phi(h)\), linear class-mean shift в kernel coordinates, kernel-weighted preimage и возврат residual | — | не начато | — | — | — | — |
| 18 | Conditional field / UniSteer | Учить \(u_\theta(z_t,t,c)\); pooled frozen condition embedding и timestep управляют SwiGLU через FiLM | — | не начато | — | — | — | — |
| 19 | INNSteer | RealNVP affine coupling blocks \(\phi\) задают обратимые coordinates; linear shift в них после \(\phi^{-1}\) становится nonlinear и input-dependent | — | не начато | — | — | — | — |
| 20 | GLP prior during training | Учить direct steering function по property loss, GLP energy и minimal-movement penalty; на inference GLP не вызывается | — | не начато | — | — | — | — |
| 21 | Финальное сравнение | Все прошедшие screening методы на одном протоколе: latency, Pareto, trajectories и failure cases | [HTML](viz.py) | не начато | — | — | — | — |

## Диагностика

![Единая геометрическая карта методов](assets/steering-geometry-concept.png)

Картинка объясняет координаты, но не является результатом. `runs/mac_screening.html` строится по реальным held-out активациям и объединяет:

- clean, steered и repaired activation, direction и correction;
- denoiser input, output, residual и промежуточные flow states;
- одну сохранённую held-out проекцию для всех methods и checkpoints, рядом с full-dimensional distances;
- train/validation loss, learning rate, gradient norm и checkpoint progression;
- local neighbours, tangent/normal split и singular spectrum только в geometry experiments;
- Jacobian spectrum только при проверке local contraction или amplification;
- gradients только при проверке alignment конкретного path с quality/property objective;
- downstream drift, logits и decoded generations на фиксированных prompts;
- causal alpha sweep с naive, repair и отключением отдельных correction components;
- Pareto, latency, memory и разобранные failure cases со ссылками на JSON, log и checkpoint.

Проверяемые geometry-гипотезы:

- random noise должен сначала увеличивать kNN distance, local-PCA residual и correction energy, а затем NLL; отсутствие такого порядка опровергает эти величины как раннюю диагностику;
- tangent/normal split должен отделить property effect от quality damage; если Pareto full, tangent и normal совпадает, локальная линейная геометрия здесь ничего не объясняет;
- nearest, Segment-kNN и local geodesic должны уменьшать full-dimensional geometry deviations при сохранении property; уменьшение только проекции PCA не считается;
- safe correction должен сохранять steering coordinate лучше обычного repair при сопоставимой geometry; рост property ценой прежнего NLL не считается улучшением;
- Curveball, UniSteer и INNSteer проверяются в собственных координатах, но сравниваются только по общим decoded Pareto и geometry axes.
