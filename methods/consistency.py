"""Неудачный вариант: consistency distillation по траектории шума."""

import models

def build(args, nets, bank, v, alpha):
    net = nets.get("consistency")
    if net is None:
        return []
    return [(f"consistency_t{t:g}", lambda h, t=t: models.repair(
        "consistency", net, h, t_start=t, steps=1)[0]) for t in args.t_start]
