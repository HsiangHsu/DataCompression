import numpy as np
from random import sample

def lloyd_max_quantizer(data, M):
    data.sort()
    A = sorted(sample(set(data), M))
    mse = 2**64

    iter_c = 0
    while True:
        B = lm_calculate_B(A)
        A = lm_calculate_A(data, B, M)
        if A == -1:
            return -1, -1, -1
        quantized = quantize_sequence(data, A, B)
        old_mse = mse
        mse = calculate_mse(data, quantized)
        iter_c += 1
        if (old_mse - mse) < 1e-5:
            return A, B, mse

def quantize(orig, A, B):
    M = len(A)
    for j in range(M):
        if (j == 0 and orig <= B[j]) or \
           (j == M-1 and B[j-1] < orig) or \
           (B[j-1] < orig <= B[j]):
            return A[j]

def quantize_sequence(orig, A, B):
    quantized = orig.copy()
    for i in range(len(orig)):
        quantized[i] = quantize(orig[i], A, B)
    return quantized

def calculate_mse(orig, quantized):
    return np.square(np.subtract(orig, quantized)).mean()

def lm_calculate_A(orig, B, M):
    A = [None] * M
    for j in range(M):
        if j == 0:
            A[j] = int(round(np.mean([x for x in orig if x <= B[j]])))
        elif j == M-1:
            A[j] = int(round(np.mean([x for x in orig if B[j-1] < x])))
        else:
            try:
                A[j] = int(round(np.mean([x for x in orig if B[j-1] < x <= B[j]])))
            except:
                return -1
    return A

def lm_calculate_B(A):
    return [(A[j] + A[j+1])/2 for j in range(0, len(A)-1)]
