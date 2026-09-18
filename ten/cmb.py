import tencore

from ten import dtype, storage_format, storage_order, tensor, diagonal

"""
Factorial n!=1x2x...xn
"""
def factorial(n, data_type = dtype.uint64):
    if data_type == dtype.uint64:
        return tencore.factorial_uint64(n)
    else:
        raise RuntimeError("Data type not supported.")

"""
Permutation of k elements from n
"""
def perm(n, k, data_type = dtype.uint64):
    if data_type == dtype.uint64:
        return tencore.perm_uint64(n, k)
    else:
        raise RuntimeError("Data type not supported.")

"""
Combination / Binomial coefficients of k from n
n!/k!(n-k)! = n(n-1)...(n-k+1)/k!
"""
def comb(n, k, data_type = dtype.uint64):
    if data_type == dtype.uint64:
        return tencore.comb_uint64(n, k)
    else:
        raise RuntimeError("Data type not supported.")

"""
Pascal triangle
"""
def pascal_triangle(n, data_type = dtype.uint64) -> tensor:
    if data_type == dtype.uint64:
        x = tencore.pascal_triangle_uint64(n)
        return tensor(x.shape(), data_type, x.format(), x.storage_order(), x)
    else:
        raise RuntimeError("Data type not supported.")

"""
nth row of pascal triangle
"""
def nth_pascal_triangle(n, data_type = dtype.uint64) -> tensor:
    if data_type == dtype.uint64:
        x = tencore.nth_pascal_triangle_uint64(n)
        return tensor(x.shape(), data_type, x.format(), x.storage_order(), x)
    else:
        raise RuntimeError("Data type not supported.")

