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


"""
Brownian motion
"""
def brownian_motion(s : tensor, t : int, data_type = dtype.float32) -> tensor:
    if data_type == dtype.float32:
        w = tencore.brownian_motion_float(s.data(), t)
        return tensor(w.shape(), data_type, w.format(), w.storage_order(), w)
    elif data_type == dtype.float64:
        w = tencore.brownian_motion_double(s.data(), t)
        return tensor(w.shape(), data_type, w.format(), w.storage_order(), w)
    else:
        raise RuntimeError("Data type not supported.")

"""
Brownian motion paths
"""
def brownian_motion_paths(n : int, t : int, s : tensor = None, data_type = dtype.float32) -> tensor:
    if s is None:
        if data_type == dtype.float32:
            w = tencore.brownian_motion_all_paths_float(n, t)
            return tensor(w.shape(), data_type, w.format(), w.storage_order(), w)
        elif data_type == dtype.float64:
            w = tencore.brownian_motion_all_paths_double(n, t)
            return tensor(w.shape(), data_type, w.format(), w.storage_order(), w)
        else:
            raise RuntimeError("Data type not supported.")
    else:
        assert s.rank() == 2
        if data_type == dtype.float32:
            w = tencore.brownian_motion_paths_float(s.data(), n, t)
            return tensor(w.shape(), data_type, w.format(), w.storage_order(), w)
        elif data_type == dtype.float64:
            w = tencore.brownian_motion_paths_float(s.data(), n, t)
            return tensor(w.shape(), data_type, w.format(), w.storage_order(), w)
        else:
            raise RuntimeError("Data type not supported.")

