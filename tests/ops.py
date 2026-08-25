import ten

x = ten.arange((3,3))
y = ten.arange((3,3))

print("Add")
z = x + y
print(z)

print("Sub")
z = x - y
print(z)

print("Mul")
a = ten.arange([10])
b = ten.arange([10])
z = a * b
print(z)

print("Div")
a = ten.arange((3,3), 1.)
b = ten.arange((3,3), 1.)
z = a / b
print(z)

print("Matmul")
z = x @ y
print(z)

print("Matvector mul")
v = ten.arange([3])
z = x @ v
print(z)

