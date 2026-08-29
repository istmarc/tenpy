from __future__ import absolute_import

import tencore

from tencore import data_type as dtype
from tencore import storage_format, storage_order

import numpy as np

# Reduce / fold
from functools import reduce
import operator

"""
Set random seed
"""
def set_seed(value):
    tencore.set_seed(value)

"""
Get the tensor type
"""


def _get_tensor(data_type, shape, sformat, order, data="auto"):
    if data == None:
        return None
    if data != "auto":
        assert data_type == data.data_type()
        assert data.size() == reduce(operator.mul, shape)
        return data
    if sformat == storage_format.dense:
        if data_type == dtype.float32:
            return tencore.tensor_float.make(shape, sformat, False, order)
        elif data_type == dtype.float64:
            return tencore.tensor_double.make(shape, sformat, False, order)
        else:
            raise RuntimeError("Data type not yet supported.")
    else:
        raise RuntimeError("Storage format not yet supported.")


"""
Get diagonal matrix
"""

def _get_diagonal(data_type, shape, order, data="auto"):
    if data == None:
        return None
    if data != "auto":
        assert data_type == data.data_type()
        assert data.rank() == 2
        assert shape[0] == shape[1]
        assert data.size() == shape[0]
        return data
    if data_type == dtype.float32:
        return tencore.diagonal_float.make(shape, False, order)
    elif data_type == dtype.float64:
        return tencore.diagonal_double.make(shape, False, order)
    else:
            raise RuntimeError("Data type not yet supported.")

def _make_tuple_shape(dims):
    if isinstance(dims, int):
        return tuple([dims])
    else:
        return tuple(dims)


def _from_numpy_data_type(data_type):
    if data_type == np.float32:
        return dtype.float32
    elif data_type == np.float64:
        return dtype.float64
    else:
        raise RuntimeError("Data type not supported.")


def _to_numpy_data_type(data_type):
    if data_type == dtype.float32:
        return np.float32
    elif data_type == dtype.float64:
        return np.float64
    else:
        raise RuntimeError("Data type not supported.")


def _getitem_from(t, index):
    data_type = t.data_type()
    if data_type == tencore.data_type.float32:
        return tencore.tensor_float_get(t, list(index))
    elif data_type == tencore.data_type.float64:
        return tencore.tensor_double_get(t, list(index))
    else:
        raise RuntimeError("Data type not supported.")


"""
Create a tensor from shape (rank), data type, and storage order
"""


class tensor(object):
    """
    tensor(shape,  data_type, storage_format, order, data = "auto")
    """

    def __init__(
        self,
        dims,
        data_type=dtype.float32,
        sformat=storage_format.dense,
        order=storage_order.col_major,
        data="auto",
    ):
        self.dims = _make_tuple_shape(dims)
        self.dims_rank = len(self.dims)
        self.data_type = data_type
        self.t = _get_tensor(data_type, self.dims, sformat, order, data)

    """
    row_major(shape, data_type, storage_format)
    """
    @classmethod
    def row_major(cls, dims, data_type=dtype.float32, sformat=storage_format.dense):
        return cls(dims, data_type, sformat, storage_order.row_major)

    def dtype(self):
        return self.data_type

    def data(self):
        return self.t

    def rank(self):
        return self.dims_rank

    def size(self):
        return self.t.size()

    def shape(self):
        return self.dims

    def dim(self, index):
        return self.t.dim(index)

    def strides(self):
        return tuple(self.t.strides())

    def format(self):
        return self.t.format()

    def storage_order(self):
        return self.t.storage_order()

    def __repr__(self):
        return repr(self.t)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= self.size():
                raise StopIteration()
            return self.t[index]
        else:
            _getitem_from(self.t, index)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.t.__setitem__(index, value)
        else:
            if self.data_type == tencore.data_type.float32:
                tencore.tensor_float_set(self.t, index, value)
            elif self.data_type == tencore.data_type.float64:
                tencore.tensor_double_set(self.t, index, value)
            else:
                raise RuntimeError("Data type not supported.")

    def __add__(self, other):
        data_type = self.dtype()
        assert(data_type == other.dtype())
        assert(self.storage_order() == other.storage_order())
        assert(self.rank() == other.rank())
        assert(self.shape() == other.shape())
        if data_type == dtype.float32:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.add_float(self.data(), other.data()))
        elif data_type == dtype.float64:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.add_float(self.data(), other.data()))
        else:
            raise RuntimeError("Data type not supported.")

    def __sub__(self, other):
        data_type = self.dtype()
        assert(data_type == other.dtype())
        assert(self.storage_order() == other.storage_order())
        assert(self.rank() == other.rank())
        assert(self.shape() == other.shape())
        if data_type == dtype.float32:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.sub_float(self.data(), other.data()))
        elif data_type == dtype.float64:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.sub_float(self.data(), other.data()))
        else:
            raise RuntimeError("Data type not supported.")

    def __truediv__(self, other):
        data_type = self.dtype()
        assert(data_type == other.dtype())
        assert(self.storage_order() == other.storage_order())
        assert(self.rank() == other.rank())
        assert(self.shape() == other.shape())
        if data_type == dtype.float32:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.div_float(self.data(), other.data()))
        elif data_type == dtype.float64:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.div_float(self.data(), other.data()))
        else:
            raise RuntimeError("Data type not supported.")

    """
    Elementwise multiplication
    """
    def __mul__(self, other):
        data_type = self.dtype()
        assert(data_type == other.dtype())
        assert(self.storage_order() == other.storage_order())
        assert(self.rank() == other.rank())
        assert(self.shape() == other.shape())
        if data_type == dtype.float32:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.mul_float(self.data(), other.data()))
        elif data_type == dtype.float64:
            return tensor(self.dims, self.dtype(), storage_format.dense, self.storage_order(), tencore.mul_float(self.data(), other.data()))
        else:
            raise RuntimeError("Data type not supported.")

    """
        Matrix multiplication and matrix vector multiplication
    """
    def __matmul__(self, other):
        data_type = self.dtype()
        assert(data_type == other.dtype())
        assert(self.storage_order() == other.storage_order())
        r = other.rank()
        assert(r == 1 or r == 2)
        assert(self.rank() == 2)
        assert(self.dim(1) == other.dim(0))
        if r == 1:
            new_shape = [other.size()]
        elif r == 2:
            new_shape = (self.dim(0), other.dim(1))
        if data_type == dtype.float32:
            return tensor(new_shape, data_type, storage_format.dense, self.storage_order(), tencore.mul_float(self.data(), other.data()))
        elif data_type == dtype.float64:
            return tensor(new_shape, data_type, storage_format.dense, self.storage_order(), tencore.mul_float(self.data(), other.data()))
        else:
            raise RuntimeError("Data type not supported.")

    """
    Convert to numpy ndarray
    """

    def to_numpy(self):
        np_data_type = _to_numpy_data_type(self.data_type)
        # FIXME Uninitialized numpy array
        array = np.zeros(self.dims, dtype=np_data_type)
        size = self.t.size()
        if self.t.storage_order() == storage_order.row_major:
            for k in range(size):
                array[k] = self.t[k]
        else:
            # col_major
            if self.dims_rank == 1:
                for k in range(size):
                    array[k] = self.t[k]
            elif self.dims_rank == 2:
                # Matrix
                rows = self.dim(0)
                cols = self.dim(1)
                for i in range(rows):
                    for j in range(cols):
                        array[i, j] = _getitem_from(self.t, [i, j])
            elif self.dims_rank == 3:
                # 3d tensor
                I = self.dim(0)
                J = self.dim(1)
                K = self.dim(2)
                for i in range(I):
                    for j in range(J):
                        for k in range(K):
                            array[i, j, k] = _getitem_from(self.t, [i, j, k])
            elif self.dims_rank == 4:
                # 4d tensor
                I = self.dim(0)
                J = self.dim(1)
                K = self.dim(2)
                L = self.dim(3)
                for i in range(I):
                    for j in range(J):
                        for k in range(K):
                            for l in range(L):
                                array[i, j, k, l] = _getitem_from(self.t, [i, j, k, l])
            elif self.dims_rank == 5:
                I = self.dim(0)
                J = self.dim(1)
                K = self.dim(2)
                L = self.dim(3)
                M = self.dim(4)
                for i in range(I):
                    for j in range(J):
                        for k in range(K):
                            for l in range(L):
                                for m in range(M):
                                    array[i, j, k, l, m] = _getitem_from(
                                        self.t, [i, j, k, l, m]
                                    )
        return array


"""
Create a vector from shape and optional data type
"""


def vector(size, data_type=dtype.float32):
    assert isinstance(size, int)
    return tensor((size), data_type)


"""
Create a matrix from shape and optional data type
"""


def matrix(rows, cols, data_type=dtype.float32):
    assert isinstance(rows, int)
    assert isinstance(cols, int)
    return tensor((rows, cols), data_type)

"""
Diagonal matrix
"""
class diagonal(object):
    """
    diagonal(dims,  data_type, order, data = "auto")
    """

    def __init__(
        self,
        dims,
        data_type=dtype.float32,
        order=storage_order.col_major,
        data="auto",
    ):
        self.dims = _make_tuple_shape(dims)
        self.dims_rank = len(self.dims)
        assert self.dims_rank == 2
        self.data_type = data_type
        self.t = _get_diagonal(data_type, self.dims, order, data)

    @classmethod
    def row_major(cls, dims, data_type=dtype.float32):
        return cls(dims, data_type, storage_order.row_major)

    def dtype(self):
        return self.data_type

    def data(self):
        return self.t

    def rank(self):
        return self.dims_rank

    def size(self):
        return self.t.size()

    def shape(self):
        return self.dims

    def dim(self, index):
        return self.t.dim(index)

    def strides(self):
        return tuple(self.t.strides())

    def format(self):
        return self.t.format()

    def storage_order(self):
        return self.t.storage_order()

    def __repr__(self):
        return repr(self.t)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= self.size():
                raise StopIteration()
            return self.t[index]
        else:
            _getitem_from(self.t, index)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.t.__setitem__(index, value)
        else:
            if self.data_type == tencore.data_type.float32:
                tencore.tensor_float_set(self.t, index, value)
            elif self.data_type == tencore.data_type.float64:
                tencore.tensor_double_set(self.t, index, value)
            else:
                raise RuntimeError("Data type not supported.")

    """
    Convert to numpy ndarray
    By default return a 1d numpy array if dense is set to False,
    otherwise return a dense numpy array
    """

    def to_numpy(self, dense = False):
        np_data_type = _to_numpy_data_type(self.data_type)
        if dense:
            # FIXME Uninitialized numpy array
            array = np.zeros(self.dims, dtype=np_data_type)
            size = self.t.size()
            if self.t.storage_order() == storage_order.row_major:
                for k in range(size):
                    array[k] = self.t[k]
            else:
                # col_major
                # Matrix
                rows = self.dim(0)
                cols = self.dim(1)
                for i in range(rows):
                    for j in range(cols):
                        array[i, j] = _getitem_from(self.t, [i, j])
        else:
            n = self.size()
            array = np.zeros((n), dtype=np_data_type)
            for k in range(n):
                array[k] = self.t[k]
        return array



"""
Create a tensor from a numpy array
by default create a col major tensor
"""


def from_numpy(array, order=storage_order.col_major):
    shape = array.shape
    data_type = _from_numpy_data_type(array.dtype)
    t = tensor(shape, data_type, storage_format.dense, order)
    size = t.size()
    rank = len(shape)
    if order == storage_order.row_major:
        for k in range(size):
            t[k] = array[k]
    else:
        # Col major tensor
        if rank == 1:
            # Vector
            for k in range(size):
                t[k] = array[k]
        elif rank == 2:
            # Matrix
            rows = shape[0]
            cols = shape[1]
            for i in range(rows):
                for j in range(cols):
                    t[i, j] = array[i, j]
        elif rank == 3:
            # 3d tensor
            I = shape[0]
            J = shape[1]
            K = shape[2]
            for i in range(I):
                for j in range(J):
                    for k in range(K):
                        t[i, j, k] = array[i, j, k]
        elif rank == 4:
            # 4d tensor
            I = shape[0]
            J = shape[1]
            K = shape[2]
            L = shape[3]
            for i in range(I):
                for j in range(J):
                    for k in range(K):
                        for l in range(L):
                            t[i, j, k, l] = array[i, j, k, l]
        elif rank == 5:
            I = shape[0]
            J = shape[1]
            K = shape[2]
            L = shape[3]
            M = shape[4]
            for i in range(I):
                for j in range(J):
                    for k in range(K):
                        for l in range(L):
                            for m in range(M):
                                t[i, j, k, l, m] = array[i, j, k, l, m]
        else:
            raise RuntimeError(
                "Array rank not supported, only up to 5d are supported for col major tensors."
            )
    return t


"""
Returns a tensor of zeros of data_type
"""


def zeros(dims, data_type=dtype.float32, order=storage_order.col_major):
    shape = _make_tuple_shape(dims)
    sformat = storage_format.dense
    if data_type == dtype.float32:
        return tensor(
            shape, data_type, sformat, order, tencore.zeros_float(shape, order)
        )
    elif data_type == dtype.float64:
        return tensor(
            shape, data_type, sforma, order, tencore.zeros_double(shape, order)
        )
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Returns a tensor of ones of data type
"""


def ones(dims, data_type=dtype.float32, order=storage_order.col_major):
    shape = _make_tuple_shape(dims)
    sformat = storage_format.dense
    if data_type == dtype.float32:
        return tensor(
            shape, data_type, sformat, order, tencore.ones_float(shape, order)
        )
    elif data_type == dtype.float64:
        return tensor(
            shape, data_type, sformat, order, tencore.ones_double(shape, order)
        )
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Returns a tensor filled with value of data_type
"""


def fill(dims, value, data_type=dtype.float32, order=storage_order.col_major):
    shape = _make_tuple_shape(dims)
    sformat = storage_format.dense
    if data_type == dtype.float32:
        return tensor(
            shape, data_type, sformat, order, tencore.fill_float(shape, value, order)
        )
    elif data_type == dtype.float64:
        return tensor(
            shape, data_type, sformat, order, tencore.fill_double(shape, value, order)
        )
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Returns a range starting from value
"""


def arange(dims, value=0.0, data_type=dtype.float32, order=storage_order.col_major):
    shape = _make_tuple_shape(dims)
    sformat = storage_format.dense
    if data_type == dtype.float32:
        return tensor(
            shape, data_type, sformat, order, tencore.range_float(shape, value, order)
        )
    elif data_type == dtype.float64:
        return tensor(
            shape, data_type, sformat, order, tencore.range_double(shape, value, order)
        )
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Returns a linear tnnsor
"""


def linear(dims, start, stop, data_type=dtype.float32, order=storage_order.col_major):
    shape = _make_tuple_shape(dims)
    sformat = storage_format.dense
    if data_type == dtype.float32:
        return tensor(
            shape,
            data_type,
            sformat,
            order,
            tencore.linear_float(shape, start, stop, order),
        )
    elif data_type == dtype.float64:
        return tensor(
            shape,
            data_type,
            sformat,
            order,
            tencore.linear_double(shape, start, stop, order),
        )
    else:
        raise RuntimeError("Data type not yet supported.")


"""
TODO Save to a binary file
def save(ten, filename):
  rank = ten.rank()
  data_type = ten.dtype()
  if rank == 1:
    if data_type == dtype.float32:
      backend.save_vector_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_vector_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.float32:
      backend.save_matrix_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_matrix_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      backend.save_tensor3_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_tensor3_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      backend.save_tensor4_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_tensor4_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      backend.save_tensor5_float(ten.data(), filename)
    elif data_type == dtype.float64:
      backend.save_tensor5_double(ten.data(), filename)
    else:
      raise RuntimeError("Saving data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")
"""

"""
Load from binary file
def load(filename, rank, data_type = dtype.float32):
  if rank == 1:
    if data_type == dtype.float32:
      data = backend.load_vector_float(filename)
      size = data.size()
      return tensor(size, data_type, data)
    elif data_type == dtype.float64:
      data = backend.load_vector_double(filename)
      size = data.size()
      return tensor(size, data_type, data)
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")
"""

"""
Returns the minimum of a tensor
"""
def min(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        return tencore.min_float(x.data()).value()
    elif data_type == dtype.float64:
        return tencore.min_double(x.data()).value()
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the maximum of a tensor
"""
def max(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        return tencore.max_float(x.data()).value()
    elif data_type == dtype.float64:
        return tencore.max_double(x.data()).value()
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the mean of a tensor
"""
def mean(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        return tencore.mean_float(x.data()).value()
    elif data_type == dtype.float64:
        return tencore.mean_double(x.data()).value()
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Returns the sum of a tensor
"""
def sum(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        return tencore.sum_float(x.data()).value()
    elif data_type == dtype.float64:
        return tencore.sum_double(x.data()).value()
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the cumulative sum of a tensor
"""
def cum_sum(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.cum_sum_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.cum_sum_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the product of the elements of a tensor
"""
def prod(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        return tencore.prod_float(x.data()).value()
    elif data_type == dtype.float64:
        return tencore.prod_double(x.data()).value()
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the square root of a tensor
"""
def sqrt(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.sqrt_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.sqrt_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the square of a tensor
"""
def sqr(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.sqr_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.sqr_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Returns the absolute value of a tensor
"""
def abs(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.abs_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.abs_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Returns the sin of a tensor
"""
def sin(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.sin_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.sin_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")



"""
Returns the sinh of a tensor
"""
def sinh(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.sinh_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.sinh_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the asin of a tensor
"""
def asin(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.asin_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.asin_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the cos of a tensor
"""
def cos(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.cos_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.cos_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the cosh of a tensor
"""
def cosh(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.cosh_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.cosh_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the acos of a tensor
"""
def acos(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.acos_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.acos_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the tan of a tensor
"""
def tan(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.tan_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.tan_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the tanh of a tensor
"""
def tanh(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.tanh_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.tanh_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the atan of a tensor
"""
def atan(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.atan_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.atan_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the exp of a tensor
"""
def exp(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.exp_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.exp_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the log of a tensor
"""
def log(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.log_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.log_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the log10 of a tensor
"""
def log10(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.log10_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.log10_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the floor of a tensor
"""
def floor(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.floor_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.floor_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the ceil of a tensor
"""
def ceil(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.ceil_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.ceil_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")


"""
Reshape a tensor
"""
def reshape(x : tensor, shape):
    # TODO Check size of x and shape
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.reshape_float(x.data(), shape)
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.reshape_double(x.data(), shape)
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Flatten a tensor
"""
def flatten(x : tensor):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.flatten_float(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.flatten_double(x.data())
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

"""
Returns the pow of a tensor
"""
def pow(x : tensor, n):
    data_type = x.dtype()
    if data_type == dtype.float32:
        y = tencore.pow_float(x.data(), n)
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    elif data_type == dtype.float64:
        y = tencore.pow_double(x.data(), n)
        return tensor(y.shape(), data_type, y.format(), y.storage_order(), y)
    else:
        raise RuntimeError("Data type not yet supported.")

