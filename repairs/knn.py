"""Притянуть точку к ближайшей настоящей активации рядом с отрезком стиринга."""

import steering


def build(args, nets, bank, v, alpha):
    repair = steering.segment_repair(bank, v, alpha) if alpha else None
    return [("knn", repair)]
