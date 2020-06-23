'''
multinomial/entropy.py

Functions for determining bounds on and (for small parameters) the true entropy
of multinomial distributions.

Equations are taken from Kaji (2016).
'''

from math import log, pi, factorial as fac
from numpy import prod, e as E


def HUa(n, P, Q):
    assert len(P) == len(Q)
    m = len(P)
    a = (1/2)*log((2*pi*n*E)**(m-1)*prod(Q))
    b = n*sum([P[i]*log(P[i]/Q[i]) for i in range(m)])
    c = sum([gU1(n, P[i], Q[i]) + (Q[i]-P[i])*gU2(n, P[i]) for i in range(m)])
    print(f'a: {a}, b: {b}, c: {c}')
    return a-b+c

def HLa(n, P, Q):
    assert len(P) == len(Q)
    m = len(P)
    a = (1/2)*log((2*pi*n*E)**(m-1)*prod(Q))
    b = 1 / (12*n)
    c = n*sum([P[i]*log(P[i]/Q[i]) for i in range(m)])
    d = sum([gL1(n, P[i], Q[i]) + (Q[i]-P[i])*gL2(n, P[i]) for i in range(m)])
    return a-b-c+d

def HUb(n, P):
    m = len(P)
    a = (1/2)*log((2*pi*n*E)**(m-1)*prod(P))
    b = sum([gU1(n, P[i], P[i]) for i in range(m)])
    return a+b

def HLb(n, P):
    m = len(P)
    a = (1/2)*log((2*pi*n*E)**(m-1)*prod(P))
    b = 1 / (12*n)
    c = sum([gL1(n, P[i], P[i]) for i in range(m)])
    return a-b+c

def gU1(n, p, q):
    a = (8*q**2 - 15*q + 8) / (12*n*q)
    b = (24*q**3 - 55*q**2 + 41*q - 10) / (12*n**2*q**2)
    c = (3*q**2 - 5*q + 2) / (6*n**3*q**2)
    d = (1-p)**n*((1/2)*log(2*pi*n*q) - (5/6) + (1/(3*n*q)))
    e = n*p*(1-p)**(n-1)*((1/2)*log(2*pi*n*q) + log(n*q) - (11/3) + \
        (9/(2*n*q)) - (9/(4*n**2*q**2)) + (1/(2*n**3*q**3)))
    return a-b-c-d-e

def gU2(n, p):
    a = (3*n**4*p**3) / 64
    b = (n**3*p**2*(54*p + 5)) / 192
    c = (n**2*p*(99*p**2 - 39*p + 106)) / 192
    d = (n*(27*p**3 - 76*p**2 + 97*p - 82)) / 96
    e = (27*p**2 - 76*p + 73) / 48
    f = (27*p - 76) / (24*n)
    g = 9 / (4*n**2)
    return -a+b-c+d+e+f+g

def gL1(n, p, q):
    a = (4*q**2 - 9*q + 5) / (12*n*q)
    b = (2*q**2 - 3*q + 1) / (6*n**2*q**2)
    c = (1-p)**n*((1/2)*log(2*pi*n*q) - ((n*q)/3) - (11/12))
    d = n*p*(1-p)**(n-1)*((1/2)*log(2*pi*n*q) + log(n*q) - ((n*q)/3) - \
        (29/12) + (5/(2*n*q)) - (11/(12*n**2*q**2)) + (1/(6*n**3*q**3)))
    return -a+b-c-d

def gL2(n, p):
    a = (n**3*p**2) / 48
    b = (n**2*p*(3*p + 10)) / (48*n)
    c = (n*(p**2 + 3*p - 23)) / 24
    d = (p - 3) / 12
    e = 1 / (6*n)
    return a-b+c+d+e


def trueH(n, P):
    m = len(P)
    a = log(fac(n))
    b = n*sum([P[i]*log(P[i]) for i in range(m)])
    c = sum([sum([pifunc(n, P[i], j)*log(fac(j)) for j in range(2, n+1)]) \
        for i in range(m)])
    return -a-b+c

def pifunc(n, p, j):
    a = fac(n) / (fac(j)*fac(n-j))
    b = p**j*(1-p)**(n-j)
    return a*b
