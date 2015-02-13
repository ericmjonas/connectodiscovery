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
    bins = np.linspace(0, 1.0, 30)
    ax.hist(assigndf['ari'], bins, normed=True)

    fname = infile[0].split('-')[0]
    ax.set_title(fname)

    ax.set_xlim(0, 1)
    f.savefig(outfile)

@transform(td("retina.*.samples.organized/assign.pickle"),
           suffix("assign.pickle"), "z.pdf")

def plot_z_matrix(exp_results, 
                  out_filename):
    assigndf = pickle.load(open(infile, 'r'))['df']


    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    exp = sample_d['exp']
    print "exp=", exp
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))

    meta_infile = meta['infile']

    d = pickle.load(open(meta_infile, 'r'))
    conn = d['conn_mat']
    cells = d['cells']

    cell_types = cells['type_id']

    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    # compute the z matrix 
    z_mat = np.zeros((len(cells), len(cells)))
    for ci, c in enumerate(chains):
        sample_latent = c['state']
        cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])
        ca = irm.util.canonicalize_assignment(cell_assignment)
        for u in np.unique(ca):
            ent_i = np.argwhere(ca == u).flatten()
            for ci in ent_i:
                for cj in ent_i:
                    z_mat[ci, cj] += 1

    import scipy.cluster
    l = scipy.cluster.hierarchy.linkage(z_mat, method='ward')
    
    
    ca = np.array(scipy.cluster.hierarchy.leaves_list(l))
    z_mat = z_mat[ca]
    z_mat = z_mat[:, ca]


    import matplotlib.gridspec as grd
    f = pylab.figure(figsize=(8, 6))
    gs = grd.GridSpec(1, 2, width_ratios=[2,10 ], wspace=0.02)
    ax = f.add_subplot(gs[:, 1])
    im = ax.imshow(z_mat/CHAINN, cmap=pylab.cm.Greys, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Cell coassignment probability") 
    ax.set_xlabel("cells")


    cbar_ax = f.add_axes([0.30, 0.15, 0.1, 0.02])
    f.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[0.0, 1.0])


    typeax = f.add_subplot(gs[:, 0])
    cells['coarse'][cells['type_id'] == 58] = 'bc'
    cmap = {None : 4, 
            'other': 4, 
            'nac' : 2, 
            'bc' : 1, 
            'mwac' : 3, 
            'gc' : 0}
    color_idx = [cmap[o] for o in cells['coarse']]        
    import brewer2mpl
    cmap = brewer2mpl.get_map('Set1', 'qualitative', 5).mpl_colormap
    typeax.scatter(cell_types, np.argsort(ca), edgecolor='none', 
               s=3, 
               c=color_idx, 
               alpha=0.8,
               cmap = cmap)
    p = np.array([0, 12, 24, 57, 72, 80])
    typeax.set_xticks(p)
    typeax.set_xticks(p[1:] - np.diff(p)/2. , minor=True)
    typeax.set_yticks([])
    typeax.set_xticklabels(['gc', 'bc', 'nac', 'mwac', 'other'], minor=True,
                           fontsize=8, rotation=90)
    typeax.set_xticklabels([])

    typeax.set_title("anatomist type (0-72)", fontsize=8)
    typeax.set_ylabel("cells")
    typeax.grid()

    # create colors
    
    typeax.set_xlim(0, 80)
    typeax.set_ylim(950, 0)


    # con = sqlite3.connect(RETINA_DB)
    # MAX_CONTACT_AREA=5.0
    # area_thold_min = 0.1
    
    # contacts_df = pandas.io.sql.read_frame("select * from contacts where area < %f and area > %f" % (MAX_CONTACT_AREA, area_thold_min), 
    #                                        con, index_col='id')

    # contacts_df.head()


    # for i in range(4):
    #     # pick points 
    #     which_row = np.argsort(ca).flatten()
    #     spos = [270, 580, 790, 930][i]
        
    #     #spos = np.argsort(ca).flatten()[spos]
    #     cell_row = np.argwhere(which_row == spos)[0]
    #     # np.argsort(ca).flatten()[spos]

    #     cell_row = cells.irow(cell_row)
    #     print cell_row
    #     cell_id = cell_row.index.values[0]
    #     print "cell_id=", cell_id
    #     typeax.scatter([cell_row['type_id']], 
    #                    [spos], c='k', s=20, edgecolor='k', 
    #                    facecolor='none')

    #     ax = f.add_subplot(gs[i, 0])
    #     c = contacts_df[contacts_df['from_id'] ==cell_id]
    #     ax.scatter(c['y'], c['x'], edgecolor='none', s=1, alpha=0.5, c='k')

    #     ax.scatter(cell_row['y'], cell_row['x'], s=40, c='r', edgecolor='none')


    #     ax.set_xlim(0, 120)
    #     ax.set_ylim(130, 50)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # print cells.head()
    f.savefig(out_filename)

if __name__ == "__main__":    
    pipeline_run([
        plot_predlinks_roc, 
        plot_ari, 
        plot_z_matrix
              ]) # , multiprocess=3)
    
