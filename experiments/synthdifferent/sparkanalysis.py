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


WORKING_DIR = "sparkdata"
S3_BUCKET = "jonas-testbucket2"
S3_PATH= "netmotifs/paper/experiments/synthdifferent"




    
def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

def get_dataset(data_name):
    return glob(td("%s.data" %  data_name))




                  
@transform(td("*.samples.organized/predlinks.pickle"),
           suffix("predlinks.pickle"), "roc.pdf")

def plot_predlinks_roc(infile, outfile):
    preddf = pickle.load(open(infile, 'r'))['df']

    preddf['tp'] = preddf['t_t'] / preddf['t_tot']
    preddf['fp'] = preddf['f_t'] / preddf['f_tot']
    preddf['frac_wrong'] = 1.0 - (preddf['t_t'] + preddf['f_f']) / (preddf['t_tot'] + preddf['f_tot'])

    f = pylab.figure(figsize=(2, 2))
    ax = f.add_subplot(1, 1, 1)
    
    # group by cv set
    for row_name, cv_df in preddf.groupby('cv_idx'):
        cv_df_m = cv_df.groupby('pred_thold').mean().sort('fp')
        ax.plot(cv_df_m['fp'], cv_df_m['tp'] , c='k', alpha=0.3)
    

    fname = infile[0].split('-')[0]
    ax.set_title(fname)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    f.savefig(outfile)

    pylab.close(f)

# @collate("sparkdata/*.samples.organized",
#          regex(r"data/(.+)-(.+).(\d\d)(.cv.*).samples"),
#          [td(r"\1-\2\4.predlinks"), td(r"\1-\2\4.assign")])
# def cv_collate_predlinks_assign(infiles_samples, (predlinks_outfile,
#                                                assign_outfile)):
    
    


@transform(td("*.samples.organized/assign.pickle"),
           suffix("assign.pickle"), "ari.pdf")
def plot_ari(infile, outfile):
    assigndf = pickle.load(open(infile, 'r'))['df']
    
    assigndf['ari'] = assigndf.apply(lambda x : metrics.adjusted_rand_score(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)

    
    f = pylab.figure(figsize=(4, 4))
    ax = f.add_subplot(1, 1, 1)
    
    bins = np.linspace(0, 1.0, 31)
    ax.hist(assigndf['ari'], bins, normed=True)

    fname = infile[0].split('-')[0]
    ax.set_title(fname)

    ax.set_xlim(0, 1)
    f.savefig(outfile)

    pylab.close(f)



def extract_metadata(fname):
    # extract metadata from the filename
    dataset_name = os.path.basename(os.path.dirname(fname))[:-len('.samples.organized')]
    fields = dataset_name.split('-')

    # skip the stupid non-replicated ones
    if "_" not in fields[0]:
        return {}

    source_model, rep_id = fields[0].split("_")
    meta = {}
    meta['source_model'] = source_model
    meta['rep_id'] = rep_id

    meta['cv_type'] = fields[1]
    meta['init_config'] = fields[2]
    meta['kernel_config'] = fields[3]

    return meta
@merge(td("*.samples.organized/assign.pickle"),
         "aggregate.stats.pickle")
def aggregate_stats(infiles, outfile):
    """
    Combine all the aggstats into a single file
    
    Compute summary statistics
    """

    res = []
    for infile in infiles:
        d = pickle.load(open(infile, 'r'))

        assigndf = d['df']

        m = extract_metadata(infile)
        if len(m) == 0:
            # skip the stupid non-replicated ones
            continue 

        for k, v in m.iteritems():
            assigndf[k] = v
        
        # compute the statistics
        assigndf['ari'] = assigndf.apply(lambda x : metrics.adjusted_rand_score(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)

        assigndf['homogeneity'] = assigndf.apply(lambda x : metrics.homogeneity(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)

        assigndf['completeness'] = assigndf.apply(lambda x : metrics.completeness(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)



        assigndf['type_n_true'] = assigndf.apply(lambda x : len(np.unique(x['true_assign'])), axis=1)
        assigndf['type_n_learned'] = assigndf.apply(lambda x : len(np.unique(x['assign'])), axis=1)
        assigndf['auc'] = assigndf.apply(lambda x: metrics.roc_auc_score(x['heldout_link_truth'], x['heldout_link_predprob']))
        assigndf['auc'] = assigndf.apply(lambda x: metrics.f1_score(x['heldout_link_truth'], x['heldout_link_predprob']))

        # 

        # fraction of mass in top N types
        
        res.append(assigndf)
    alldf = pandas.concat(res)
    pickle.dump(alldf, open(outfile, 'w'), -1)
        


if __name__ == "__main__":

    pipeline_run([                  
        plot_predlinks_roc,
        plot_ari, 
        aggregate_stats
        
              ])
    
    
