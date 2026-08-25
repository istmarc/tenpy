import tencore

from ten import dtype, tensor

"""
Markov Chain Monte Carlo (MCMC)
"""
def mcmc(xt, n, f, g = None, burn = 0, dtype = dtype.float32):
    if g == None:
        if dtype == dtype.float32:
            r = tencore.mcmc2_float(xt, n, f, burn)
            return tensor(r.shape(), dtype, r.format(), r.storage_order(), r)
        elif dytep == dtype.float64:
            r = tencore.mcmc2_double(xt, n, f, burn)
            return tensor(r.shape(), dtype, r.format(), r.storage_order(), r)
        else:
            raise RuntimeError("Data type not supported.")
    else:
        if dtype == dtype.float32:
            r = tencore.mcmc_float(xt, n, f, g, burn)
            return tensor(r.shape(), dtype, r.format(), r.storage_order(), r)
        elif dytep == dtype.float64:
            r = tencore.mcmc_double(xt, n, f, g, burn)
            return tensor(r.shape(), dtype, r.format(), r.storage_order(), r)
        else:
            raise RuntimeError("Data type not supported.")

