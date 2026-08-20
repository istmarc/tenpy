from __future__ import absolute_import

import tencore

from tencore import data_type as dtype
from tencore import storage_format, storage_order

import numpy as np

# Reduce / fold
from functools import reduce
import operator

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
        return tencore.tensor_float_get(t, index)
    elif data_type == tencore.data_type.float64:
        return tencore.tensor_double_get(t, index)
    else:
        raise RuntimeError("Data type not supported.")


"""
Create a tensor from shape (rank), data type, and storage order
"""


class tensor(object):
    """
    tensor(dims,  data_type, data = "auto")
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
def min(x:tensor):
  rank = x.rank()
  data_type = x.dtype()
  if rank == 1:
    if data_type == dtype.float32:
      return backend.min_vector_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_vector_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.float32:
      return backend.min_matrix_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_matrix_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      return backend.min_tensor3_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_tensor3_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      return backend.min_tensor4_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_tensor4_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      return backend.min_tensor5_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.min_tensor5_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")
"""

"""
Returns the maximum of a tensor
def max(x:tensor):
  rank = x.rank()
  data_type = x.dtype()
  if rank == 1:
    if data_type == dtype.float32:
      return backend.max_vector_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_vector_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 2:
    if data_type == dtype.float32:
      return backend.max_matrix_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_matrix_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 3:
    if data_type == dtype.float32:
      return backend.max_tensor3_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_tensor3_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 4:
    if data_type == dtype.float32:
      return backend.max_tensor4_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_tensor4_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  elif rank == 5:
    if data_type == dtype.float32:
      return backend.max_tensor5_float(x.data()).eval().value()
    elif data_type == dtype.float64:
      return backend.max_tensor5_double(x.data()).eval().value()
    else:
      raise RuntimeError("Saving data type not yet supported.")
  else:
    raise RuntimeError(f"Tensor of rank {rank} not supported.")
"""
