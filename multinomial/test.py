#!/usr/bin/env python3

import argparse
from math import log
import numpy as np

from entropy import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, required=True)
parser.add_argument('-P', type=float, nargs='+', required=True)
parser.add_argument('-H', action='store_true')
args = parser.parse_args()

n = args.n
P = args.P
calc_H = args.H
m = len(P)
Q = [max(P[i], 2/n) for i in range(m)]

if (np.array(P) == np.array(Q)).all():
    U = HUb(n, P)
    L = HLb(n, P)
else:
    U = HUa(n, P, Q)
    L = HLa(n, P, Q)


print(f'n: {n}')
print(f'Upper: {U} nats, {U/log(2)} bits')
if calc_H:
    H = trueH(n, P)
    print(f'H: {H} nats, {H/log(2)} bits')
print(f'Lower: {L} nats, {L/log(2)} bits')
