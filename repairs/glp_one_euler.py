"""Тот же velocity field, что у GLP, но один шаг Эйлера вместо двадцати."""

import denoiser


def build(args, nets, bank, v, alpha):
    return [(f"glp1_t{t:g}", lambda h, t=t: denoiser.sdedit_onestep(nets["glp"], h, t))
            for t in args.t_start]
