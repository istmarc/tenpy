import tencore

from ten import dtype, storage_format, storage_order, tensor, diagonal

"""
Random walk with a uniform distribution by default
"""
def random_walk(n, k = 0., prob = .5, inc = 1., data_type = dtype.float32, prob_type = dtype.float32):
    if data_type == dtype.float32:
        if prob_type == dtype.float32:
            s = tencore.random_walk_float_float(n, k, prob, inc)
            return tensor(s.shape(), data_type, s.format(), s.storage_order(), s)
        else:
            raise RuntimeError("Probabilty type not supported.")
    elif data_type == dtype.float64:
        if prob_type == dtype.float64:
            s = tencore.random_walk_double_double(n, k, prob, inc)
            return tensor(s.shape(), data_type, s.format(), s.storage_order(), s)
        else:
            raise RuntimeError("Probabilty type not supported.")
    else:
        raise RuntimeError("Unsupported data type")


