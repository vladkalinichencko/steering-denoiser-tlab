"""Базовый метод задания: одношаговый denoiser, обученный на MSE."""


def build(args, nets, bank, v, alpha):
    return [("mse", lambda h: nets["mse"].repair(h))]
