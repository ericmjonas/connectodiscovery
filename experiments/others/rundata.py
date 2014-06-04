import numpy as np
import cPickle as pickle
import sqlite3
import sys
import others
import pandas
from ruffus import * 


sys.path.append("../connattrib")

import preprocess


AREA_THOLD_MIN = 0.1
dbname = "../preprocess/mouseretina/mouseretina.db"

@files(None, "data/data.pickle")
def create_data(infile, outfile):

    conn = sqlite3.connect(dbname)
    cells, conn_mat, dist_mats = preprocess.create_data(conn, AREA_THOLD_MIN)
    dist_xyz = np.sqrt(dist_mats['x']**2 + dist_mats['y']**2 + dist_mats['z']**2)
    dist_yz = np.sqrt(dist_mats['y']**2 + dist_mats['z']**2)

    conn_mat = conn_mat.astype(np.float)

    have_edges_i = (conn_mat.sum(axis=1) > 0)
    conn_mat_have_edges = conn_mat[have_edges_i]
    conn_mat_have_edges = conn_mat_have_edges[:, have_edges_i]

    pickle.dump({'conn_mat' : conn_mat, 
                 'cells' : cells, 'dist_mats' : dist_mats, 
                 'dist_yz' : dist_yz, 
                 'dist_xyz' : dist_xyz, 
                 'have_edges_i' : have_edges_i, 
                 'conn_mat_have_edges' : conn_mat_have_edges}, 
                open(outfile, 'w'))

def vblpcm_params():
    for G in [3, 5, 15, 20, 30]:
        for d in [2, 3, 4, 5]:
            infile = 'data/data.pickle'
            outfile = 'data/data.vblpcm.%d.%d.pickle' % (G, d)
            yield infile, outfile, G, d

@follows(create_data)
@files(vblpcm_params)
def vblpcm_cluster(infile, outfile, G, d):
    data = pickle.load(open(infile, 'r'))
    conn_mat_have_edges = data['conn_mat_have_edges']
    
    res = others.vblpcm_cluster(conn_mat_have_edges, G, d, 
                                STEPS=1000, maxiter=100)

    pickle.dump(res, 
                open(outfile, 'w'))

@merge(vblpcm_cluster, 'vblpcm.results.pickle')
def vblpcm_cluster_merge(infiles, outfile):
    res = []
    for f in infiles:
        f_attrib = f.split('.')
        d = int(f_attrib[-2])
        G = int(f_attrib[-3])
        a = pickle.load(open(f, 'r'))

        res.append({'d' : d, 'G' : G, 
                    'assignments' : a['assignments'], 
                    'auc' : a['roc'], 
                    'conv' : a['conv']})
    df = pandas.DataFrame(res)
    
    pickle.dump(df, open(outfile, 'w'))

if __name__ == "__main__":
    pipeline_run([vblpcm_cluster, vblpcm_cluster_merge])


