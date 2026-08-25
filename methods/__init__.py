"""Один файл на способ починки активации; eval_steering только выбирает нужные."""

from methods import (
    consistency,
    curveball,
    glp,
    glp_one_euler,
    inn,
    knn,
    meanflow,
    mse,
    none,
    rectified,
    tangent_mse,
    unisteer,
)

BUILDERS = {
    "none": none.build,
    "mse": mse.build,
    "knn": knn.build,
    "glp": glp.build,
    "glp1": glp_one_euler.build,
}

FAILED_BUILDERS = {
    "tangent_mse": tangent_mse.build,
    "consistency": consistency.build,
    "meanflow": meanflow.build,
    "rectified": rectified.build,
    "curveball": curveball.build,
    "inn": inn.build,
    "unisteer": unisteer.build,
}
