"""
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
from sklearn import metrics
import models
import multyvac

import irm
import irm.data
import util
from irm import rand
import predutil
import boto

import cvpipelineutil as cv
import sparkutil


def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


def to_f32(x):
    a = np.array(x).astype(np.float32)
    assert type(a) == np.ndarray
    return a


WORKING_DIR = "sparkdatacv"
S3_BUCKET = "jonas-testbucket2"





    
def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

def get_dataset(data_name):
    return glob(td("%s.data" %  data_name))


def extract_metadata(fname):
    # extract metadata from the filename
    dataset_name = os.path.basename(os.path.dirname(fname))[:-len('.samples.organized')]
    fields = dataset_name.split('-')

    # skip the stupid non-replicated ones
    data_name, rel_status, source_model, rep_id = fields[0].split(".")
    meta = {}
    meta['source_model'] = source_model
    meta['rep_id'] = rep_id

    meta['cv_type'] = fields[1]
    meta['init_config'] = fields[2]
    meta['kernel_config'] = fields[3]

    return meta

@merge(td("*fixed_20_200*.samples.organized/aggstats.pickle"),
         "aggregate.stats.pickle")
def aggregate_stats(infiles, outfile):
    """
    Combine all the aggstats into a single file
    
    Compute summary statistics
    """

    res = []
    for infile in infiles:
        d = pickle.load(open(infile, 'r'))
        print "The file is", infile
        assigndf = d['df']
        meta = d['meta']
        neurons = meta['neurons']


        m = extract_metadata(infile)
        if len(m) == 0:
            # skip the stupid non-replicated ones
            continue 

        for k, v in m.iteritems():
            assigndf[k] = v
        

        assigndf['true_assign_role'] = [np.array(neurons['role']) for _ in range(len(assigndf))]
        # compute the statistics
        assigndf['ari'] = assigndf.apply(lambda x : metrics.adjusted_rand_score(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)

        assigndf['homogeneity'] = assigndf.apply(lambda x : metrics.homogeneity_score(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)

        assigndf['completeness'] = assigndf.apply(lambda x : metrics.completeness_score(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)


        # don't consider the ones where the role is "none" as these are multi-role ones
        neurons.ix[neurons['role'].isnull(), 'role'] = 'I'
        
        assigndf['role_ari'] = assigndf.apply(lambda x : metrics.adjusted_rand_score(neurons['role'], 
                                                                                     irm.util.canonicalize_assignment(x['assign'])), axis=1)

        assigndf['role_homogeneity'] = assigndf.apply(lambda x : metrics.homogeneity_score(neurons['role'], 
                                                                                           irm.util.canonicalize_assignment(x['assign'])), axis=1)

        assigndf['role_completeness'] = assigndf.apply(lambda x : metrics.completeness_score(neurons['role'], 
                                                                                             irm.util.canonicalize_assignment(x['assign'])), axis=1)



        assigndf['type_n_true'] = assigndf.apply(lambda x : len(np.unique(x['true_assign'])), axis=1)
        assigndf['type_n_learned'] = assigndf.apply(lambda x : len(np.unique(x['assign'])), axis=1)
        assigndf['auc'] = assigndf.apply(lambda x: metrics.roc_auc_score(x['heldout_link_truth'], x['heldout_link_predprob']), axis=1)
        #assigndf['f1'] = assigndf.apply(lambda x: metrics.f1_score(x['heldout_link_truth'], x['heldout_link_predprob']), axis=1)

        # 

        # fraction of mass in top N types
        
        res.append(assigndf)
    alldf = pandas.concat(res)
    pickle.dump(alldf, open(outfile, 'w'), -1)
        


if __name__ == "__main__":

    pipeline_run([                  

        aggregate_stats
        
              ])
    
    
