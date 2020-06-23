#!/usr/bin/env python3

from math import log
import numpy as np

from entropy import *

n = 200
m = 3
P = [0.01, 0.01, 0.98]
Q = [max(P[i], 2/n) for i in range(m)]

if (np.array(P) == np.array(Q)).all():
    U = HUb(n, P)
    L = HLb(n, P)
else:
    U = HUa(n, P, Q)
    L = HLa(n, P, Q)

H = trueH(n, P)

print(f'n: {n}')
print(f'Upper: {U} nats, {U/log(2)} bits')
print(f'H: {H} nats, {H/log(2)} bits')
print(f'Lower: {L} nats, {L/log(2)} bits')
