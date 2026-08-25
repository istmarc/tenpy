import ten

ten.set_seed(123)

unif = ten.random.uniform(0., 1., ten.dtype.float32)

for i in range(2):
    print(unif.sample())

x = unif.sample((2,2), ten.storage_order.col_major)
print(x)
print(x.to_numpy())

norm = ten.random.normal(0., 1., ten.dtype.float64)
x = norm.sample((2,2))
print(x)
print(x.to_numpy())

x = ten.random.rand_unif((2, 2))
print(x)
print(x.to_numpy())

x = ten.random.rand_norm((2,2))
print(x)
print(x.to_numpy())

