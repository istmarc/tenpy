import tencore

from tencore import graph_type
from ten import dtype, tensor

def _get_glist(vertex_type, gtype):
    if vertex_type == dtype.int32:
        return tencore.glist_int32(gtype)
    elif vertex_type == dtype.uint32:
        return tencore.glist_uint32(gtype)
    elif vertex_type == dtype.int64:
        return tencore.glist_int64(gtype)
    elif vertex_type == dtype.uint64:
        return tencore.glist_uint64(gtype)
    elif vertex_type == dtype.string:
        return tencore.glist_str(gtype)

def _get_gedge(vertex_type, gtype):
    if vertex_type == dtype.int32:
        return tencore.gedge_int32(gtype)
    elif vertex_type == dtype.uint32:
        return tencore.gedge_uint32(gtype)
    elif vertex_type == dtype.int64:
        return tencore.gedge_int64(gtype)
    elif vertex_type == dtype.uint64:
        return tencore.gedge_uint64(gtype)
    elif vertex_type == dtype.string:
        return tencore.gedge_str(gtype)

def _get_gmatrix(vertices, vertex_type, gtype):
    if vertex_type == dtype.float32:
        return tencore.gmatrix_float(vertices, gtype)
    elif vertex_type == dtype.float64:
        return tencore.gmatrix_double(vertices, gtype)

def _get_gweighted(vertex_type, weight_type, gtype):
    if vertex_type == dtype.int32 and weight_type == dtype.int32:
        return tencore.gweighted_int32_int32(gtype)
    elif vertex_type == dtype.int32 and weight_type == dtype.float32:
        return tencore.gweighted_int32_float(gtype)
    elif vertex_type == dtype.int32 and weight_type == dtype.float64:
        return tencore.gweighted_int32_double(gtype)
    elif vertex_type == dtype.int64 and weight_type == dtype.int64:
        return tencore.gweighted_int64_int64(gtype)
    elif vertex_type == dtype.int64 and weight_type == dtype.float32:
        return tencore.gweighted_int64_float(gtype)
    elif vertex_type == dtype.int64 and weight_type == dtype.float64:
        return tencore.gweighted_int64_double(gtype)
    elif vertex_type == dtype.uint32 and weight_type == dtype.uint32:
        return tencore.gweighted_uint32_uint32(gtype)
    elif vertex_type == dtype.uint32 and weight_type == dtype.float32:
        return tencore.gweighted_uint32_float(gtype)
    elif vertex_type == dtype.uint32 and weight_type == dtype.float64:
        return tencore.gweighted_uint32_double(gtype)
    elif vertex_type == dtype.uint64 and weight_type == dtype.uint64:
        return tencore.gweighted_uint64_uint64(gtype)
    elif vertex_type == dtype.uint64 and weight_type == dtype.float32:
        return tencore.gweighted_uint64_float(gtype)
    elif vertex_type == dtype.uint64 and weight_type == dtype.float64:
        return tencore.gweighted_uint64_double(gtype)
    elif vertex_type == dtype.string and weight_type == dtype.float32:
        return tencore.gweighted_str_float(gtype)
    elif vertex_type == dtype.string and weight_type == dtype.float64:
        return tencore.gweighted_str_double(gtype)

"""
Undirected/Directed Graph data structure.
It can be a list, an edge list, a matrix or a weighted list.
"""
class graph(object):
    def __init__(self, gtype = graph_type.undirected, g = None):
        self.gtype = gtype
        self.g = g
        if self.g is None:
            self.g = tencore.glist_uint64(self.gtype)

    @classmethod
    def glist(cls, gtype = graph_type.undirected, vertex_type = dtype.uint64):
        return cls(gtype, _get_glist(vertex_type, gtype))

    @classmethod
    def gedge(cls, gtype = graph_type.undirected, vertex_type = dtype.uint64):
        return cls(gtype, _get_gedge(vertex_type, gtype))

    @classmethod
    def gmatrix(cls, vertices, gtype = graph_type.undirected, vertex_type = dtype.float32):
        return cls(gtype, _get_gmatrix(vertices, vertex_type, gtype))

    @classmethod
    def gweighted(cls, gtype = graph_type.undirected, vertex_type = dtype.uint64, weight_type = dtype.float32):
        return cls(gtype, _get_gweighted(vertex_type, weight_type, gtype))

    def empty(self):
        return self.g.empty()

    def add_vertex(self, vertex):
        self.g.add_vertex(vertex)

    def add_edge(self, source, dest):
        self.g.add_edge(source, dest)

    def has_edge(self, source, dest):
        self.g.has_edge(source, dest)

    def to_matrix(self, data_type = dtype.float32):
        if data_type == dtype.float32:
            m = self.g.to_matrix_float()
            return tensor(m.shape(), m.data_type(), m.format(), m.storage_order(), m)
        elif data_type == dtype.float64:
            m = self.g.to_matrix_double()
            return tensor(m.shape(), m.data_type(), m.format(), m.storage_order(), m)
        elif data_type == dtype.int32:
            m = self.g.to_matrix_int32()
            return tensor(m.shape(), m.data_type(), m.format(), m.storage_order(), m)
        elif data_type == dtype.uint32:
            m = self.g.to_matrix_uint32()
            return tensor(m.shape(), m.data_type(), m.format(), m.storage_order(), m)
        elif data_type == dtype.int64:
            m = self.g.to_matrix_int64()
            return tensor(m.shape(), m.data_type(), m.format(), m.storage_order(), m)
        elif data_type == dtype.uint64:
            m = self.g.to_matrix_uint64()
            return tensor(m.shape(), m.data_type(), m.format(), m.storage_order(), m)

    def dfs(self, source, fn):
        self.g.dfs(source, fn)

    def bfs(self, source, fn):
        self.g.bfs(source, fn)

    def to_networkx(self):
        pass

