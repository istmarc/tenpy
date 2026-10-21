import tencore

from ten import dtype, tensor

"""
Euler's method
"""
def euler(t0, y0, df, h, n, data_type = dtype.float32):
    if data_type == dtype.float32:
        return tencore.euler_float(t0, y0, df, h, n)
    elif data_type == dtype.float64:
        return tencore.euler_double(t0, y0, df, h, n)

