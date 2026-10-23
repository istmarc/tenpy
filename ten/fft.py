import tencore

from ten import dtype, tensor, to_tensor

"""
FFT
"""
def fft(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.fft_float(x.data())
        return to_tensor(y)
    elif data_type == dtype.float64:
        y = tencore.fft_double(x.data())
        return to_tensor(y)


