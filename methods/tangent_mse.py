"""Неудачный вариант: MSE только вдоль локального касательного подпространства."""

import models

def build(args, nets, bank, v, alpha):
    net = nets.get("tangent_mse")
    if net is None:
        return []
    return [("tangent_mse", lambda h: models.repair("tangent_mse", net, h)[0])]
