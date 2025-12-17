#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple random search within a bounded hypercube

@author: Rodolphe Le Riche, Brian Dédji Whannou
"""
import numpy as np
from optim_utilities import record_best
from optim_utilities import record_any


def random_opt(func: object, LB, UB, budget: int = 1e3, printlevel=0):
    if len(LB) != len(UB):
        raise Exception(
            "the bounds should have the same nb of components and it is %d and %d"
            % (len(LB), len(UB))
        )
    dim = len(UB)
    iteration = 0
    f_best = float("inf")
    x_best = np.zeros(dim)
    condition = False
    res = {}  # initialize a dict for optimization results

    while not condition:
        xnew = np.random.uniform(low=LB, high=UB)
        fnew = func(xnew)
        iteration += 1
        res = record_any(rec=res, f=fnew, x=xnew, time=iteration, printlevel=printlevel)
        if fnew < f_best:
            f_best = fnew
            x_best = xnew
            res = record_best(
                rec=res,
                fbest=f_best,
                xbest=x_best,
                time=iteration,
                printlevel=printlevel,
            )

        condition = iteration >= budget

    res["stop_condition"] = "budget exhausted "
    return res
