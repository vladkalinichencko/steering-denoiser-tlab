"""GLP: возврат на многообразие за двадцать шагов Эйлера.

t_start здесь главный рычаг: он решает, сколько от правки остаётся и сколько чинится,
поэтому каждое его значение становится отдельной точкой.
"""

import denoiser


def build(args, nets, bank, v, alpha):
    return [(f"glp_t{t:g}", lambda h, t=t: denoiser.sdedit(nets["glp"], h, t, args.steps))
            for t in args.t_start]
