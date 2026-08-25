import ten
ten.set_seed(123)

x = ten.random.rand_unif([10], -1., 1.)
print(x)

print(ten.sort(x))

ten.sort_inplace(x)
print(x)

