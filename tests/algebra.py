import ten

ten.set_seed(123)

x = ten.arange((3,3))

print("X = ")
print(x)

print("QR factorization")
q,r = ten.linalg.qr(x)
print(q)
print(r)
print(q.to_numpy() @ r.to_numpy())

print("LU factorization")
p, l,u = ten.linalg.lu(x)
print(p)
print(l)
print(u)

print("Cholesky factorization")
l,u = ten.linalg.cholesky(x)
print(l)
print(u)

print("SVD factorization")
x = ten.random.rand_norm((3, 3))
print(x)
u, s, vt = ten.linalg.svd(x)
print(u)
print(s)
print(vt)

