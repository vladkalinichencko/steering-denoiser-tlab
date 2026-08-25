"""Один файл на способ починки активации; eval_steering только выбирает нужные."""

from repairs import glp, glp_one_euler, knn, mse, none

BUILDERS = {
    "none": none.build,
    "mse": mse.build,
    "knn": knn.build,
    "glp": glp.build,
    "glp1": glp_one_euler.build,
}
