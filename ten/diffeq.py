import tencore

from ten import dtype, tensor, to_tensor

"""
Euler's method
"""
def euler(t0, y0, df, h, n, data_type = dtype.float32):
    if data_type == dtype.float32:
        t,y = tencore.euler_float(t0, y0, df, h, n)
        return to_tensor(t), to_tensor(y)
    elif data_type == dtype.float64:
        t,y = tencore.euler_double(t0, y0, df, h, n)
        return to_tensor(t), to_tensor(y)

