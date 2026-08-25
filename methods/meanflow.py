"""Неудачный вариант: MeanFlow с JVP-регуляризацией."""

import models

def build(args, nets, bank, v, alpha):
    net = nets.get("meanflow")
    if net is None:
        return []
    return [(f"meanflow_t{t:g}", lambda h, t=t: models.repair(
        "meanflow", net, h, t_start=t, steps=1)[0]) for t in args.t_start]
