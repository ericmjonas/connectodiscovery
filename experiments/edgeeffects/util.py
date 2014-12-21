import numpy as np
import sklearn.metrics


def compute_conn_hist(cell_pos, connectivity, type_info, TYPE_N, DIST_BIN_EDGES):

    dist = sklearn.metrics.euclidean_distances(cell_pos, cell_pos)
    #iui = np.triu_indices(len(dist), 1)
    #d_ul = dist[iui[0], iui[1]]
    #assert len(d_ul) == (len(dist)) * (len(dist) -1)/2 

    CELL_N = len(cell_pos)
    DIST_BIN_N = len(DIST_BIN_EDGES)
    dist_binned = np.searchsorted(DIST_BIN_EDGES[1:], dist)
    assert dist_binned.shape == (CELL_N, CELL_N)
    conn_hist = np.zeros((TYPE_N, TYPE_N, 2, DIST_BIN_N), dtype=np.float32)
    for ci in range(CELL_N):
        for cj in range(CELL_N):
            ci_type = type_info[ci]
            cj_type = type_info[cj]

            dist_bin = dist_binned[ci, cj]
            conn = connectivity[ci, cj]
            if conn:
                conn_i = 1
            else:
                conn_i = 0 
            conn_hist[ci_type, cj_type, conn_i, dist_bin] += 1
    return conn_hist
