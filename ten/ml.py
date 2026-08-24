import tencore

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

