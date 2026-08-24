import tencore

from ten import dtype, storage_format, storage_order, tensor, diagonal

"""
QR factorization
"""
def qr(a : tensor):
    data_type = a.dtype()
    sformat = a.format()
    order = a.storage_order()
    if data_type == dtype.float32:
        q,r = tencore.qr_float(a.data())
        return tensor(q.shape(), data_type, sformat, order, q), tensor(r.shape(), data_type, sformat, order, r)
    elif data_type  == dtype.float64:
        q,r = tencore.qr_double(a.data())
        return tensor(q.shape(), data_type, sformat, order, q), tensor(r.shape(), data_type, sformat, order, r)
    else:
        raise RuntimeError("Data type not supported.")

"""
LU factorization
"""
def lu(a : tensor):
    data_type = a.dtype()
    sformat = a.format()
    order = a.storage_order()
    if data_type == dtype.float32:
        p, l,u = tencore.lu_float(a.data())
        return tensor(p.shape(), data_type, sformat, order, p), tensor(l.shape(), data_type, sformat, order, l), tensor(u.shape(), data_type, sformat, order, u)
    elif data_type  == dtype.float64:
        p, l,u = tencore.lu_double(a.data())
        return tensor(p.shape(), data_type, sformat, order, p), tensor(l.shape(), data_type, sformat, order, l), tensor(u.shape(), data_type, sformat, order, u)
    else:
        raise RuntimeError("Data type not supported.")

"""
Cholesky factorization
"""
def cholesky(a : tensor):
    data_type = a.dtype()
    sformat = a.format()
    order = a.storage_order()
    if data_type == dtype.float32:
        l,u = tencore.cholesky_float(a.data())
        return tensor(l.shape(), data_type, sformat, order, l), tensor(u.shape(), data_type, sformat, order, u)
    elif data_type  == dtype.float64:
        l,u = tencore.cholesky_double(a.data())
        return tensor(l.shape(), data_type, sformat, order, l), tensor(u.shape(), data_type, sformat, order, u)
    else:
        raise RuntimeError("Data type not supported.")

"""
SVD factorization
"""
def svd(a : tensor):
    data_type = a.dtype()
    sformat = a.format()
    order = a.storage_order()
    if data_type == dtype.float32:
        u, s, vt = tencore.svd_float(a.data())
        return tensor(u.shape(), data_type, sformat, order, u), diagonal(s.shape(), data_type, order, s), tensor(vt.shape(), data_type, sformat, order, vt)
    elif data_type  == dtype.float64:
        u, s, vt = tencore.svd_double(a.data())
        return tensor(u.shape(), data_type, sformat, order, u), diagonal(s.shape(), data_type, order, s), tensor(vt.shape(), data_type, sformat, order, vt)
    else:
        raise RuntimeError("Data type not supported.")

"""
Least squares, solve Ax=b
"""
def solve(A: tensor, b : tensor, method = "qr"):
    assert isinstance(A, tensor)
    assert isinstance(b, tensor)
    assert A.dtype() == b.dtype()
    assert method == "qr" or method == "lu" or method == "svd"
    data_type = A.dtype()
    if data_type == dtype.float32:
        if method == "qr":
            return tencore.lsqr_float(A.data(), b.data())
        elif method == "lu":
            return tencore.lslu_float(A.data(), b.data())
        else:
            return tencore.lssvd_float(A.data(), b.data())
    elif data_type == dtype.float64:
        if method == "qr":
            return tencore.lsqr_double(A.data(), b.data())
        elif method == "lu":
            return tencore.lslu_double(A.data(), b.data())
        else:
            return tencore.lssvd_double(A.data(), b.data())
    else:
        raise RuntimeError("Data type not supported.")

