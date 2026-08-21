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

## Сетап

| поле | значение |
|---|---|
| LM | GPT-2 |
| активация | `blocks.6.hook_resid_post`, размерность 768 |
| train data | FineWeb, все позиции кроме BOS, document-level train/validation split |
| validation directions | заранее фиксированные held-out SAE directions и contrastive directions |
| область интервенции | только генерируемые response tokens; prompt и специальные токены не меняются |
| strength | относительная шкала по средней норме активации, включая нулевую точку |
| quality | conditional continuation NLL/perplexity под чистой GPT-2 |
| property | scorer, зафиксированный вместе с каждым validation direction |
| локальная проверка | MPS, один seed, не больше 30 минут на метод |
| полный запуск | A100, общий evaluation path, финальные методы с повторными seeds |

## Эксперименты

| эксперимент | идея | код | статус | данные | результат | диагностика |
|---|---|---|---|---|---|---|
| 1. Чистый cache активаций | Обучать priors на естественных non-BOS состояниях и делить документы, а не отдельные токены | [сбор](train_denoiser.py) | повторить | — | — | noise tolerance, density, spectrum |
| 2. Валидация steering directions | До обучения проверить, что direction вызывает заявленное свойство в генерациях; encoder activations и decoder effect смотреть отдельно | [baseline](baseline.py) | выбрать held-out набор | [старый SAE pilot](logs/check_sae27677.json), [второй latent](logs/pareto_sae4875.json) | Для двух проверенных latents encoder response не совпал с decoder steering effect | tokens, logits, generations, alpha ablation |
| 3. Толерантность к случайному шуму | Найти масштаб, с которого isotropic perturbation меняет качество и внутреннюю динамику | — | не начато | — | — | quality по sigma, trajectories, downstream drift |
| 4. Наивный линейный steering | Общая точка отсчёта без repair | [baseline](baseline.py), [evaluation](eval_steering.py) | повторить на новом path | [старый pilot](logs/check_diffmeansentiment.json) | — | Pareto, generations, tensor path |
| 5. GLP + SDEdit | Flow matching учит распределение активаций; многократное reverse integration возвращает steered state в вероятную область | [model](denoiser.py), [train](train_denoiser.py), [eval](eval_steering.py) | повторить | [старый train log](logs/glp_history.jsonl), [старый Pareto](logs/pareto_sentiment.json) | — | flow loss, generated/real geometry, denoising trajectory |
| 6. MSE denoiser | Сравнить additive Gaussian и interpolation corruption в простой MLP и в общем с GLP backbone без timestep conditioning | [model](denoiser.py), [train](train_denoiser.py) | interpolation повторить; остальные варианты не запущены | [старый train log](logs/mse_history.jsonl) | — | clean reconstruction, repair residual, Pareto |
| 7. Дообученный MLP GPT-2 | Использовать существующий MLP среднего слоя как denoiser без новой inference-сети | [model](denoiser.py), [train](train_denoiser.py) | код есть, не запущен | — | — | weight drift, clean/steered behavior, Pareto |
| 8. Tangent-preserving denoiser | Сохранять локальную variation вдоль tangent basis и удалять normal corruption; safe-вариант не удаляет компонент correction вдоль direction | [train](train_denoiser.py), [repair](steering.py) | код есть, не запущен | — | — | local basis, tangent/normal residual, causal ablations |
| 9. Selective repair | Применять denoiser только там, где согласованный departure score показывает повреждение | — | обсудить score до кода | [старый gated pilot](runs/steering_gated.json) | — | score calibration, repaired-token map, generations |
| 10. Ближайшие реальные активации | Возвращать state к activation bank: nearest neighbour или ближайшая точка рядом с steering segment | [repair](steering.py) | код segment repair есть, не запущен | — | — | neighbour panel, distance, copied-state failures |
| 11. Tangent/normal decomposition | Разложить direction в локальном basis и причинно проверить, где находятся property effect и quality damage | [split](steering.py), [eval](eval_steering.py) | повторить с non-BOS local basis | [старый global-PCA pilot](logs/split_sentiment.json) | — | full/tangent/normal Pareto, local residual, spectra |
| 12. Локальная геодезическая | Идти маленькими шагами и после каждого шага пересчитывать local tangent direction | [prototype](steering.py) | heuristic-код есть, не запущен | — | — | curved trajectory, step utility, density |
| 13. Curveball Steering | Делать нелинейную интервенцию в polynomial-kernel PCA coordinates и возвращаться через preimage | — | не начато | — | — | distortion, latent path, reconstruction error |
| 14. Одношаговый Euler GLP | Один раз вычислить instantaneous velocity; дешёвый контроль, не MeanFlow | [repair](denoiser.py), [eval](eval_steering.py) | код есть, не запущен | — | — | endpoint error против 20-step GLP, latency, Pareto |
| 15. Дистилляция GLP | Учить one-step student воспроизводить endpoints многократного GLP; consistency-вариант требует согласованности по разным noise levels | — | не начато | — | — | teacher/student trajectories, endpoint error, latency |
| 16. Rectified Flow | Выпрямлять transport paths, чтобы coarse или one-step integration меньше ошибалась | — | не начато | — | — | path curvature, one/few-step error, Pareto |
| 17. MeanFlow | Учить average velocity между текущим и конечным временем и сразу предсказывать полный one-step displacement | — | не начато | — | — | average/instantaneous velocity, endpoint error, Pareto |
| 18. Conditional nonlinear field | Учить input-dependent direction по state и текстовому условию; UniSteer-вариант делает conditional flow inversion | — | не начато | — | — | field trajectories, condition ablation, composition tests |
| 19. INNSteer | Учить invertible coordinates, где property отделяется линейно; обратное отображение даёт нелинейный input-dependent path | — | не начато | — | — | invertibility error, latent/original trajectories, Pareto |
| 20. Prior как training constraint | Использовать GLP energy только во время обучения прямого steering function или LM MLP, без repair на inference | — | не начато | — | — | train-time energy, inference trajectory, no-prior ablation |
| 21. Финальное сравнение | Все прошедшие baseline-проверку методы на одних directions, prompts, seeds и alpha scale | [HTML](viz.py) | не начато | — | — | общий HTML с Pareto, trajectories и failure gallery |

## Диагностика

- **Pareto и тексты:** property/quality с интервалами, все decoded generations и failure cases; [HTML](viz.py).
- **Noise tolerance:** random-noise strength против quality, property и decoded output; код —.
- **Tensor path:** clean activation, steered activation, repaired activation и correction residual; код —.
- **Траектории:** alpha, denoising steps и checkpoints в одной PCA-системе, обученной на held-out real activations; [черновик](tmp/diag_denoise_path.py).
- **Локальная геометрия:** kNN cloud, local PCA residual, tangent/normal decomposition и curvature; код —.
- **Density и prior:** kNN distance, GLP reconstruction/energy и distance до real activation bank; [черновик](tmp/diag_steering.py).
- **Propagation:** изменение activation по последующим слоям и изменение logits/softmax; [черновик](tmp/diag_steering.py).
- **Gradients и Jacobian:** alignment loss-gradient с steering/correction и local singular values; код —.
- **Rank и spectrum:** effective rank token-by-dimension и singular-value curves по слоям и checkpoints; код —.
- **Training dynamics:** train/validation objective, gradient norm, learning rate и fixed-example trajectories по checkpoints; код —.
- **Causal ablations:** zero alpha, shuffled/orthogonal direction, repair off, tangent/normal и preserved-direction correction; код —.
