import tencore

from ten import dtype, storage_format, storage_order, tensor

# Distributions
class dist(object):
    def __init__(self, d, data_type):
        self.d = d
        self.data_type = data_type

    """
    sample() to get a sampled value
    sample(dims [,order]) to get a sampled vector of shape dims
    """
    def sample(self, *args):
        if len(args) == 0:
            return self.d.sample()
        elif len(args) == 1:
            dims = args[0]
            return tensor(dims, self.data_type, storage_format.dense, storage_order.col_major, self.d.sample_tensor(dims, storage_order.col_major))
        elif len(args) == 2:
            dims = args[0]
            order = args[1]
            return tensor(dims, self.data_type, storage_format.dense, order, self.d.sample_tensor(dims, order))
        else:
            raise RuntimeError("Unsupported argument type.")

# Uniform
def uniform(lower_bound = 0., upper_bound = 1., data_type = dtype.float32):
    if data_type == dtype.float32:
        return dist(tencore.uniform_float.make(lower_bound, upper_bound), data_type)
    elif data_type == dtype.float64:
        return dist(tencore.uniform_double.make(lower_bound, upper_bound), data_type)
    else:
        raise RuntimeError("Unsupported data type.")

# Normal
def normal(mean = 0., std = 1., data_type = dtype.float32):
    if data_type == dtype.float32:
        return dist(tencore.normal_float.make(mean, std), data_type)
    elif data_type == dtype.float64:
        return dist(tencore.normal_double.make(mean, std), data_type)
    else:
        raise RuntimeError("Unsupported data type.")

# Gamma
def gamma(alpha, beta, data_type = dtype.float32):
    if data_type == dtype.float32:
        return dist(tencore.gamma_float.make(alpha, beta), data_type)
    elif data_type == dtype.float64:
        return dist(tencore.gamma_double.make(alpha, beta), data_type)
    else:
        raise RuntimeError("Unsupported data type.")

# Random uniform
def rand_unif(dims, lower_bound = 0., upper_bound = 1., data_type = dtype.float32,
    order = storage_order.col_major):
    assert (data_type == dtype.float32 or data_type == dtype.float64)
    if data_type == dtype.float32:
        return tensor(dims, data_type, storage_format.dense, order, tencore.rand_unif_float(dims, lower_bound, upper_bound, order))
    elif data_type == dtype.float64:
        return tensor(dims, data_type, storage_format.dense, order, tencore.rand_unif_double(dims, lower_bound, upper_bound, order))
    else:
        raise RuntimeError("Data type not supported.")

# Random normal
def rand_norm(dims, mean = 0., std = 1., data_type = dtype.float32,
    order = storage_order.col_major):
    assert (data_type == dtype.float32 or data_type == dtype.float64)
    if data_type == dtype.float32:
        return tensor(dims, data_type, storage_format.dense, order, tencore.rand_norm_float(dims, mean, std, order))
    elif data_type == dtype.float64:
        return tensor(dims, data_type, storage_format.dense, order, tencore.rand_norm_double(dims, mean, std, order))
    else:
        raise RuntimeError("Data type not supported.")

