"""
SPARK_DRIVER_MEMORY=4g ~/projects/sparktest/src/spark-1.1.0-bin-cdh4/bin/spark-submit  --conf spark.kryoserializer.buffer.mb=512 --conf spark.akka.frameSize=1000  --conf spark.executor.memory=4g --conf spark.python.worker.memory=4g --master local[2]

"""
import sys
import matplotlib
matplotlib.use('Agg')
# total hack, I should really know better
sys.path.append("../../code")

from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os, glob, sys, shutil
from glob import glob
import time
from matplotlib import pylab
import matplotlib
import pandas
import colorbrewer
from nose.tools import assert_equal 
from sklearn import metrics
import preprocess
import sqlite3
import connattribio 

import matplotlib.gridspec as gridspec
import models

import irm
import irm.data
import util
from irm import rand
import boto
import sklearn.metrics

WORKING_DIR = "data"

    
def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

def nodist_params():
    for i in range(10):
        yield None, td("sbmnodist.%02d.sourcedata" % i), i

@files(nodist_params)
def create_data_sbm_nodist(_, outfile, seed):
    np.random.seed(seed)
    conn_config = {}
    CLASS_N = 4
    SIDE_N = 20
    nonzero_frac = 0.8
    JITTER = 0.5


    for c1 in range(CLASS_N):
        for c2 in range(CLASS_N):
            if np.random.rand() < nonzero_frac:
                conn_config[(c1, c2)] = (1000, # guarantee that we always have conn
                                         np.random.uniform(0.1, 0.9))

    nodes_with_meta, connectivity = irm.data.generate.c_class_neighbors(SIDE_N, 
                                                                        conn_config,
                                                                        JITTER=JITTER)


    
    ai = np.random.permutation(len(nodes_with_meta))
    nodes_with_meta = nodes_with_meta[ai]
    connectivity = connectivity[ai]
    connectivity = connectivity[:, ai]
    
    pickle.dump({'nodes_with_meta' : nodes_with_meta,
                 'connectivity' : connectivity,
                 'conn_config' : conn_config,
                 'jitter' : JITTER},
                open(outfile, 'w'))


def srm_params():
    for i in range(10):
        yield None, td("srm.%02d.sourcedata" % i), i

@files(srm_params)
def create_data_srm(infile, outfile, seed):
    np.random.seed(seed)
    conn_config = {}
    CLASS_N = 4
    SIDE_N = 20
    nonzero_frac = 0.8
    JITTER = 0.5

    for c1 in range(CLASS_N):
        for c2 in range(CLASS_N):
            if np.random.rand() < nonzero_frac:
                conn_config[(c1, c2)] = (np.random.uniform(0.5, 30.0), 
                                         np.random.uniform(0.1, 0.9))
    if len(conn_config) == 0:
        conn_config[(0, 0)] = (np.random.uniform(1.0, 4.0), 
                               np.random.uniform(0.4, 0.9))

    nodes_with_meta, connectivity = irm.data.generate.c_class_neighbors(SIDE_N, 
                                                                        conn_config,
                                                                        JITTER=JITTER)


    
    ai = np.random.permutation(len(nodes_with_meta))
    nodes_with_meta = nodes_with_meta[ai]
    connectivity = connectivity[ai]
    connectivity = connectivity[:, ai]
    
    pickle.dump({'nodes_with_meta' : nodes_with_meta,
                 'connectivity' : connectivity,
                 'conn_config' : conn_config,
                 'jitter' : JITTER},
                open(outfile, 'w'))


@transform([create_data_srm, create_data_sbm_nodist],
           suffix(".sourcedata"), ".var")
def compute_variance(infile, outfile):
    d = pickle.load(open(infile, 'r'))
    nodes_with_meta = d['nodes_with_meta']
    cell_pos = nodes_with_meta['pos']
    print nodes_with_meta.dtype
    connectivity = d['connectivity']
    conn_config = d['conn_config']


    
    # look at the variance
    CENTER_N = 10
    MAX_D = 28
    TYPE_N = len(np.unique(nodes_with_meta['class']))
    assert TYPE_N == (np.max(nodes_with_meta['class'])+1)
    DIST_BIN_N = 30
    DIST_BINS = np.linspace(0, MAX_D, DIST_BIN_N)    


    res = []
    
    for center_i in range(CENTER_N):
        center = np.random.uniform(0, MAX_D, 3)
        center[2] = 0.5 # Z is always in-plane
        


        for radius in np.linspace(1, 20, 20):
            dist = sklearn.metrics.euclidean_distances(cell_pos, center).flatten()

            region_radius = radius
            sel_points = np.argwhere(dist < region_radius).flatten()
            subset_pos = cell_pos[sel_points]
            subset_conn = connectivity[sel_points]
            subset_conn = subset_conn[:, sel_points]
            subset_type = nodes_with_meta['class'][sel_points]
            conn_hist = util.compute_conn_hist(subset_pos, subset_conn, subset_type, TYPE_N, DIST_BINS)


            for ti in range(TYPE_N):
                for tj in range(TYPE_N):

                    if (ti, tj) in conn_config:
                        thold, p = conn_config[(ti, tj)]

                    h = conn_hist[ti, tj, :, :].sum(axis=0)
                    p = conn_hist[ti, tj, 1, :] / h
                    var= np.var(p[h > 0])
                    res.append({'ti' : ti, 'tj' : tj, 
                                'var' : var, 'radius' : radius,
                                'center' : center,
                                'center_i' : center_i})

    df = pandas.DataFrame(res)
    pickle.dump(df, open(outfile, 'w'), -1)

    
if __name__ == "__main__":
    pipeline_run([create_data_srm, create_data_sbm_nodist,
                  compute_variance], multiprocess=4)
    
    
