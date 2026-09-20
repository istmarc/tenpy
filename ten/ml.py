import tencore

from tencore import lm_method
from ten import dtype, tensor

import matplotlib.pyplot as plt

"""
Histogram
"""
class histogram(object):
    def __init__(self, data_type = dtype.float32, cv = False, standartize = False, cumulative = False, nbins = 0):
        self.data_type = data_type
        options = tencore.histogram_options(cv, standartize, cumulative, nbins)
        self.h = None
        if data_type == dtype.float32:
            self.h = tencore.histogram_float(options)
        elif data_type == dtype.float64:
            self.h = tencore.histogram_double(options)
        else:
            raise RuntimeError("Data type not supported.")

    def fit(self, data):
        assert isinstance(data, tensor)
        assert data.rank() == 1
        assert data.dtype() == self.data_type
        self.h.fit(data.data())

    def hist(self):
        hist, bins = self.h.hist()
        return tensor(hist.shape(), hist.data_type(), hist.format(), hist.storage_order(), hist), tensor(bins.shape(), bins.data_type(), bins.format(), bins.storage_order(), bins)

    def plot(self, color = "black", fill = "black", **args):
        hist, bins = self.hist()
        fig, ax = plt.subplots()
        ax.stairs(hist.to_numpy(), bins.to_numpy(), color = color, fill = fill, **args)

"""
Plot a histogram
"""
def hist(x: tensor, cv = False, standartize = False, cumulative = False, nbins = 0, color = "black", fill = "black", **args):
    h = histogram(x.dtype(), cv, standartize, cumulative, nbins)
    h.fit(x)
    h.plot(color = color, fill = fill, **args)

def _get_linear_model(data_type, method):
    if data_type == dtype.float32:
        return tencore.linear_model_float(method)
    elif data_type == dtype.float64:
        return tencore.linear_model_double(method)
    else:
        raise RuntimeError("Data type not supported.")

"""
Linear model
"""
class linear_model(object):
    def __init__(self, method = tencore.lm_method.qr, data_type = dtype.float32):
        self.data_type = data_type
        self.model = _get_linear_model(data_type, method)

    def fit(self, x, y, fitted = True):
        assert isinstance(x, tensor)
        assert isinstance(y, tensor)
        self.model.fit(x.data(), y.data(), fitted)

    def coef(self):
        beta = self.model.coef()
        return tensor(beta.shape(), self.data_type, beta.format(), beta.storage_order(), beta)

    def fitted():
        yhat = self.model.fitted()
        return tensor(yhat.shape(), self.data_type, yhat.format(), yhat.storage_order(), yhat)

def _get_polyreg(data_type, n):
    if data_type == dtype.float32:
        return tencore.polyreg_float(n)
    elif data_type == dtype.float64:
        return tencore.polyreg_double(n)
    else:
        raise RuntimeError("Data type not supported.")

"""
Polynomial regression
"""
class polyreg(object):
    def __init__(self, n, data_type = dtype.float32):
        self.data_type = data_type
        self.n = n
        self.model = _get_polyreg(data_type, n)

    def fit(self, x, y, fitted = True):
        assert isinstance(x, tensor)
        assert isinstance(y, tensor)
        self.model.fit(x.data(), y.data(), fitted)

    def coef(self):
        beta = self.model.coef()
        return tensor(beta.shape(), self.data_type, beta.format(), beta.storage_order(), beta)

    def fitted(self):
        yhat = self.model.fitted()
        return tensor(yhat.shape(), self.data_type, yhat.format(), yhat.storage_order(), yhat)

