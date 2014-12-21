"""
"""
import sys
# total hack, I should really know better
sys.path.append("../../code")
import matplotlib
matplotlib.use('Agg')

from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os, glob, sys
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
import multyvac

import matplotlib.gridspec as gridspec
import models

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

DEFAULT_CORES = 8
DEFAULT_RELATION = "ParRelation"
WORKING_DIR = "sparkdatacv"
S3_BUCKET = "jonas-testbucket2"
S3_PATH= "netmotifs/paper/experiments/celegans"




def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)



@transform(td("celegans.*.samples.organized/predlinks.pickle"),
           suffix("predlinks.pickle"), "roc.pdf")
def plot_predlinks_roc(infile, outfile):
    print "infile =", infile
    preddf = pickle.load(open(infile, 'r'))['df']
    preddf['tp'] = preddf['t_t'] / preddf['t_tot']
    preddf['fp'] = preddf['f_t'] / preddf['f_tot']
    preddf['frac_wrong'] = 1.0 - (preddf['t_t'] + preddf['f_f']) / (preddf['t_tot'] + preddf['f_tot'])

    f = pylab.figure(figsize=(4, 4))
    ax = f.add_subplot(1, 1, 1)
    
    # group by cv set
    for row_name, cv_df in preddf.groupby('cv_idx'):
        cv_df_m = cv_df.groupby('pred_thold').mean().sort('fp')
        ax.plot(cv_df_m['fp'], cv_df_m['tp'], alpha=0.5 )

    fname = infile[0].split('-')
    ax.set_title(fname)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    f.savefig(outfile)

# @collate("sparkdata/*.samples.organized",
#          regex(r"data/(.+)-(.+).(\d\d)(.cv.*).samples"),
#          [td(r"\1-\2\4.predlinks"), td(r"\1-\2\4.assign")])
# def cv_collate_predlinks_assign(infiles_samples, (predlinks_outfile,
#                                                assign_outfile)):
    
    


@transform(td("celegans.*.samples.organized/assign.pickle"),
           suffix("assign.pickle"), "ari.pdf")
def plot_ari(infile, outfile):
    assigndf = pickle.load(open(infile, 'r'))['df']
    print assigndf
    
    assigndf['ari'] = assigndf.apply(lambda x : metrics.adjusted_rand_score(x['true_assign'], irm.util.canonicalize_assignment(x['assign'])), axis=1)

    
    f = pylab.figure(figsize=(4, 4))
    ax = f.add_subplot(1, 1, 1)
    
    # # group by cv set
    # for row_name, cv_df in preddf.groupby('cv_idx'):
    #     print "PLOTTING", row_name
    #     cv_df = cv_df.sort('fp')
    #     ax.plot(cv_df['fp'], cv_df['tp'])
    bins = np.linspace(0, 1.0, 30)
    ax.hist(assigndf['ari'], bins, normed=True)

    fname = infile[0].split('-')[0]
    ax.set_title(fname)

    ax.set_xlim(0, 1)
    f.savefig(outfile)

if __name__ == "__main__":    
    pipeline_run([
        plot_predlinks_roc, 
        plot_ari
              ]) # , multiprocess=3)
    
