import tencore

from ten import dtype, tensor

"""
Inverse transform sampling
"""
def inv_sample(size, Finv, data_type = dtype.float32):
    if data_type == dtype.float32:
        x = tencore.inv_sample_float(size, Finv)
        return tensor(x.shape(), data_type, x.format(), x.storage_order(), x)
    else:
        raise RuntimeError("Data type not supported.")


