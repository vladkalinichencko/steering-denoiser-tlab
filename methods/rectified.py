"""Неудачный вариант: rectified flow с замороженным GLP-учителем."""

import models

def build(args, nets, bank, v, alpha):
    net = nets.get("rectified")
    if net is None:
        return []
    return [(f"rectified_{steps}step", lambda h, steps=steps: models.repair(
        "rectified", net, h, t_start=0.2, steps=steps)[0])
            for steps in (1, 2, 4)]
