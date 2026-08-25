import ten
ten.set_seed(123)

n = 10
A = ten.random.rand_norm((n, n))
b = ten.random.rand_norm([n])

print("QR\n")
xqr = ten.linalg.solve(A, b, "qr")
print(xqr)

print("LU\n")
xlu = ten.linalg.solve(A, b, "lu")
print(xlu)

print("SVD\n")
xsvd = ten.linalg.solve(A, b, "svd")
print(xsvd)


