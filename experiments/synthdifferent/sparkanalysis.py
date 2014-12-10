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
from nose.tools import assert_equal 
from sklearn import metrics
import preprocess
import sqlite3
import connattribio 

import matplotlib.gridspec as gridspec
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

EXPERIMENTS = [
    ('srm', 'cv_nfold_2', 'debug_2_100', 'debug_20'), 
    ('srm', 'cv_nfold_2', 'debug_2_100', 'debug_10'), 
    ('srm', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
    ('sbmnodist', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
    ('lpcm', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
    ('mm', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
    # ('lpcm', 'cx_4_5', 'debug_2_100', 'debug_200'), 
    # ('mm', 'cx_4_5', 'debug_2_100', 'debug_200'), 


    #('retina.xsoma' , 'fixed_20_100', 'anneal_slow_400'), 
]

PRED_EVALS= np.logspace(-4, 0, 41) # np.linspace(0, 1.0, 41)


INIT_CONFIGS = {'fixed_20_100' : {'N' : 20, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'debug_2_100' : {'N' : 2, 
                                 'config' : {'type' : 'fixed', 
                                             'group_num' : 100}}, 
            }

CV_CONFIGS = {'cv_nfold_10' : {'N' : 10,
                               'type' : 'nfold'},
              'cv_nfold_2' : {'N' : 2,
                               'type' : 'nfold'},
          }


slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 800



def generate_ld_hypers():
    space_vals =  irm.util.logspace(1.0, 80.0, 40)
    p_mins = np.array([0.001, 0.005, 0.01])
    p_maxs = np.array([0.99, 0.95, 0.90, 0.80])
    res = []
    for s in space_vals:
        for p_min in p_mins:
            for p_max in p_maxs:
                res.append({'lambda_hp' : s, 'mu_hp' : s, 
                           'p_min' : p_min, 'p_max' : p_max})
    return res


slow_anneal[0][1]['subkernels'][-1][1]['grids']['LogisticDistance'] = generate_ld_hypers()



KERNEL_CONFIGS = {
    'anneal_slow_1000' : {'ITERS' : 1000, 
                         'kernels' : slow_anneal},
    'debug_20' : {'ITERS' : 2, 
                  'kernels': irm.runner.default_kernel_anneal(1.0, 20)
              },
    'debug_10' : {'ITERS' : 2, 
                  'kernels': irm.runner.default_kernel_anneal(1.0, 20)
              },
    'debug_200' : {'ITERS' : 200, 
                  'kernels': irm.runner.default_kernel_anneal(1.0, 160)
              }
    }


# 


                  
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
    
    # # group by cv set
    # for row_name, cv_df in preddf.groupby('cv_idx'):
    #     print "PLOTTING", row_name
    #     cv_df = cv_df.sort('fp')
    #     ax.plot(cv_df['fp'], cv_df['tp'])
    bins = np.linspace(0, 1.0, 31)
    ax.hist(assigndf['ari'], bins, normed=True)

    fname = infile[0].split('-')[0]
    ax.set_title(fname)

    ax.set_xlim(0, 1)
    f.savefig(outfile)

if __name__ == "__main__":

    pipeline_run([                  
                  plot_predlinks_roc,
                  plot_ari, 
        
              ])
    
    
