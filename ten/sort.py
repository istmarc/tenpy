import tencore

from ten import dtype, tensor

"""
Sort a tensor
"""
def sort(x : tensor):
    assert isinstance(x, tensor)
    data_type = x.dtype()
    if data_type == dtype.float32:
        return tensor(x.shape(), data_type, x.format(), x.storage_order(), tencore.sort_float(x.data()))
    elif data_type == dtype.float64:
        return tensor(x.shape(), data_type, x.format(), x.storage_order(), tencore.sort_double(x.data()))
    else:
        raise RuntimeError("Data type not supported.")

"""
Sort inplace
"""
def sort_inplace(x : tensor):
    assert isinstance(x, tensor)
    data_type = x.dtype()
    if data_type == dtype.float32:
        tencore.sort_inplace_float(x.data())
    elif data_type == dtype.float64:
        tencore.sort_inplace_double(x.data())
    else:
        raise RuntimeError("Data type not supported.")


