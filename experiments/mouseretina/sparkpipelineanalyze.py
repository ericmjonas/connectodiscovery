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
S3_PATH= "netmotifs/paper/experiments/mouseretina"


RETINA_DB = "../../../preprocess/mouseretina/mouseretina.db"

CV_CONFIGS = {'cv_nfold_10' : {'N' : 10,
                               'type' : 'nfold'},
              'cv_nfold_2' : {'N' : 2,
                              'type' : 'nfold'}}




def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

EXPERIMENTS = [
    ('retina.0.ld.0.0.xyz', 'cv_nfold_2', 'debug_2_100', 'debug_2'), 


    # ('retina.xsoma' , 'fixed_20_100', 'anneal_slow_1000'), 
    # ('retina.1.bb' , 'fixed_20_100', 'anneal_slow_1000'), 
    #('retina.xsoma' , 'fixed_20_100', 'anneal_slow_400'), 
]


THOLDS = [0.01, 0.1, 0.5, 1.0]
    
MULAMBS = [1.0, 5.0, 10.0, 20.0, 50.0]
PMAXS = [0.95] # , 0.9, 0.7]

BB_ALPHAS = [1.0]
BB_BETAS = [1.0]

VAR_SCALES = [0.01, 0.1] # , 1.0]
COMP_KS = [2, 3]

# for ti in [1]: # remember to add 2 back in ! 
#     for v in range(len(VAR_SCALES)):
#         for k in COMP_KS:
#             EXPERIMENTS.append(('retina.%d.clist.%d.%d' % (ti, v, k) , 
#                                 'fixed_20_100', 'anneal_slow_1000'))


# # for ti in range(len(THOLDS)):
# #     for ml_i in range(len(MULAMBS)):
# #         for pmax_i in range(len(PMAXS)):
# #             for vars in ['x', 'yz', 'xyz']:
# #                 bs = 'retina.%d.ld.%d.%d.%s' % (ti, ml_i, pmax_i, vars)
# #                 EXPERIMENTS.append((bs, 'fixed_20_100', 'anneal_slow_400'))


for ti in [1]: # (len(THOLDS)):
    for ml_i in [3]:
        for pmax_i in range(len(PMAXS)):
            for vars in ['xyz']: 
                bs = 'retina.%d.ld.%d.%d.%s' % (ti, ml_i, pmax_i, vars)
                #EXPERIMENTS.append((bs, 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'))


for ti in [1]:
    for ml_i in [3] : 
        for pmax_i in range(len(PMAXS)):
            for vars in ['xyz']:
                for var_scale in range(len(VAR_SCALES)):
                    for comp_k in COMP_KS:
                        bs = 'retina.%d.srm_clist_xsoma.%d.%d.%s.%d.%d' % (ti, ml_i, pmax_i, vars, var_scale, comp_k)
                        EXPERIMENTS.append((bs, 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'))
                

            
INIT_CONFIGS = {'fixed_20_100' : {'N' : 20, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'debug_2_100' : {'N' : 2, 
                                 'config' : {'type' : 'fixed', 
                                             'group_num' : 100}}, 
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
slow_anneal[0][1]['subkernels'][-1][1]['grids']['MixtureModelDistribution'] = None


def soma_x_hp_grid():
    GRIDN = 10
    mu = np.linspace(20, 120, GRIDN+1) 
    sigmasq = irm.util.logspace(1.0, 2.0, GRIDN)
    kappa = [0.1, 1.0]
    nu = irm.util.logspace(10.0, 50.0, GRIDN) 
    
    hps = []
    for m in mu:
        for s in sigmasq:
            for k in kappa:
                for n in nu:
                    hps.append({'mu' : m, 
                                'kappa' : k, 
                                'sigmasq' : s, 
                                'nu' : n})
    return hps

slow_anneal[0][1]['subkernels'][-1][1]['grids']['r_soma_x'] = None  # soma_x_hp_grid()


KERNEL_CONFIGS = {
    'anneal_slow_1000' : {'ITERS' : 1000, 
                         'kernels' : slow_anneal},
    'debug_2' : {'ITERS' : 2, 
                  'kernels': irm.runner.default_kernel_anneal(1.0, 2)
              }
    }




@transform(td("retina.*.samples.organized/predlinks.pickle"),
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
        print "PLOTTING", row_name
        cv_df = cv_df.sort('fp')
        ax.plot(cv_df['fp'], cv_df['tp'], c='k', alpha=0.3)

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
    
    


@transform(td("retina.*.samples.organized/assign.pickle"),
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
    bins = np.linspace(0, 1.0, 11)
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
    
