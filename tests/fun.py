import ten

ten.set_seed(1234)

x = ten.rand_norm((4,4))
print(x)

print("Min")
m = ten.min(x)
print(m)

print("Max")
m = ten.max(x)
print(m)

print("Mean")
m = ten.mean(x)
print(m)

print("Sum")
s = ten.min(x)
print(s)

print("Cumulative sum")
y = ten.cum_sum(x)
print(y)

print("Prod")
p = ten.prod(x)
print(p)

print("Sqrt")
y = ten.sqrt(x)
print(y)

print("Sqr")
y = ten.sqr(x)
print(y)

print("Exp")
y = ten.exp(x)
print(y)

print("Log")
y = ten.log(x)
print(y)

print("Cos")
y = ten.cos(x)
print(y)


print("Sin")
y = ten.sin(x)
print(y)

print("Tanh")
y = ten.tan(x)
print(y)

print("Floor")
y = ten.floor(x)
print(y)

print("Ceil")
y = ten.ceil(x)
print(y)

print("Pow(2)")
y = ten.pow(x, 2)
print(y)

