import ten
import math
import random

mu = .5
sigma = 1.

def f(x):
    v = x-mu;
    return math.exp((-1. / (2 * sigma * sigma)) * (v*v))

def g(y):
    return y + random.uniform(-1., 1.)

x = ten.mcmc(0., 1000, f, g)
print(x)

y = ten.mcmc(0., 1000, f)
print(y)

