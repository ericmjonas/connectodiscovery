"""
To run locally

SPARK_DRIVER_MEMORY=4g ~/projects/sparkutil/src/spark-1.1.0-bin-cdh4/bin/spark-submit  --conf spark.kryoserializer.buffer.mb=512 --conf spark.akka.frameSize=1000  --conf spark.executor.memory=4g --conf spark.python.worker.memory=4g --master local[2] sparkpipelinecv.py


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

import matplotlib.gridspec as gridspec
import models

import irm
import irm.data
import util
from irm import rand

import predutil
import boto
from pyspark import SparkContext
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


CELEGANS_DB = "celegans_herm.db"

CV_CONFIGS = {'cv_nfold_10' : {'N' : 10,
                               'type' : 'nfold'},
              'cv_nfold_2' : {'N' : 2,
                              'type' : 'nfold'}}



INIT_CONFIGS = {'debug_2_100' : {'N' : 2, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 100}}, 
                'fixed_100_200' : {'N' : 100, 
                                  'config' : {'type' : 'fixed', 
                                              'group_num' : 200}}, 
}


def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

EXPERIMENTS = [
    ('celegans.2r.ldp.00', 'cv_nfold_2', 'debug_2_100', 'debug_2'), 


    # ('retina.xsoma' , 'fixed_20_100', 'anneal_slow_1000'), 
    # ('retina.1.bb' , 'fixed_20_100', 'anneal_slow_1000'), 
    #('retina.xsoma' , 'fixed_20_100', 'anneal_slow_400'), 
]


BB_HPS = [(0.1, 0.1)]# , (1.0, 1.0), (2.0, 2.0), (3.0, 1.0)]
GP_HPS = [(1.0, 2.0)]# , (2.0, 2.0), (3.0, 2.0)]
LD_HPS = [(0.1, 0.1, 0.9, 0.01)] # , (0.1, 0.1, 0.5, 0.001),
LDFL_HPS = [(0.2, 0.4, 0.01, 1.0, 1.0)] # , (0.1, 0.1, 0.5, 0.001),
NDFW_HPS = [(0.1, 0.001, 0.1, 1.0, 1.0)]
SDB_HPS = [(1.0, 1.0, 0.1, 0.001, 0.5, 4.0)]
EDP_HPS = [(0.1, 1.0, 1.0), (0.1, 1.0, 2.0), (0.1, 1.0, 4.0), (0.1, 1.0, 8.0) , 
           (0.1, 1.0, 6.0)]

LDP_HPS = [(1.0, 1.0, 0.01, 2.0, 1.0), 
           (1.0, 1.0, 0.01, 2.0, 2.0), 
           (1.0, 1.0, 0.01, 2.0, 4.0), 
           (1.0, 1.0, 0.01, 2.0, 8.0), 
           (1.0, 1.0, 0.01, 2.0, 6.0), 
       ]
default_nonconj = irm.runner.default_kernel_nonconj_config()
default_conj = irm.runner.default_kernel_config()
default_anneal = irm.runner.default_kernel_anneal()
slow_anneal = irm.runner.default_kernel_anneal()
slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
slow_anneal[0][1]['anneal_sched']['iterations'] = 300
slow_anneal[0][1]['subkernels'][-2][1]['grid'] = np.linspace(1.0, 50.0, 50.0)

def generate_ld_hypers():
    mu_min = 0.01
    mu_max = 0.9
    space_vals =  irm.util.logspace(mu_min, mu_max, 50)
    p_mins = np.array([0.001])
    p_maxs = np.array([0.90]) # , 0.80, 0.50, 0.20])
    res = []
    for s in space_vals:
        for p_min in p_mins:
            for p_max in p_maxs:
                res.append({'lambda_hp' : s, 'mu_hp' : s, 
                           'p_min' : p_min, 'p_max' : p_max})
    return res

slow_anneal[0][1]['subkernels'][-1][1]['grids']['LogisticDistance'] = generate_ld_hypers()

def generate_ldfl_hypers():
    space_vals =  irm.util.logspace(0.01, 0.9, 40)
    p_mins = np.array([0.001, 0.01, 0.02])
    alpha_betas = [0.1, 1.0, 2.0]
    res = []
    for s in space_vals:
        for p_min in p_mins:
            for alpha_beta in alpha_betas:
                res.append({'lambda' : s, 'mu_hp' : s, 
                            'p_min' : p_min, 
                            'p_scale_alpha_hp' : alpha_beta, 
                            'p_scale_beta_hp' : alpha_beta})
    return res


def generate_sdb_hypers():

    p_alphas = irm.util.logspace(0.001, 1.0, 20)
    p_betas = irm.util.logspace(0.5, 10.0, 20)
    mu_hps = irm.util.logspace(0.01, 0.4, 10)
    hps = []
    for a in p_alphas:
        for b in p_betas:
            for mu_hp in mu_hps:
                hps.append({'p_alpha' : a, 'p_beta': b, 
                            'mu_hp' : mu_hp, 
                            'p_min' : 0.001, 
                            'param_weight' : 0.5, 
                            'param_max_distance' : 4.0})
                
    return hps

slow_anneal[0][1]['subkernels'][-1][1]['grids']['SquareDistanceBump'] = generate_sdb_hypers()


def generate_ndfw_hypers():
    p_mins = np.array([0.001, 0.01, 0.02])
    mus = np.array([0.1, 1.0])
    widths = np.array([0.1, 0.5, 1.0])
    alphabetas = [0.1, 1.0, 2.0]
    hps = []
    for p_min in p_mins:
        for mu in mus:
            for width in widths:
                for a in alphabetas:
                    for b in alphabetas:
                        hps.append({'mu_hp' : mu, 
                                    'p_min' : p_min, 
                                    'width' : width, 
                                    'p_alpha' : a,
                                    'p_beta' : b})
    return hps

slow_anneal[0][1]['subkernels'][-1][1]['grids']['NormalDistanceFixedWidth'] = generate_ndfw_hypers()

slow_anneal[0][1]['subkernels'][-1][1]['grids']['BetaBernoulli'] =  [{'alpha' : a, 'beta' : b} for a, b in irm.util.cart_prod([irm.util.logspace(0.001, 1.0, 20), irm.util.logspace(0.1, 10, 20)])]

slow_anneal[0][1]['subkernels'][-1][1]['grids']['ExponentialDistancePoisson'] = irm.gridgibbshps.default_grid_exponential_distance_poisson(dist_scale = 2.0, rate_scale_scale = 20.0, GRIDN = 20)


slow_anneal[0][1]['subkernels'][-1][1]['grids']['LogisticDistancePoisson'] = irm.gridgibbshps.default_grid_logistic_distance_poisson(dist_scale = 2.0, rate_scale_scale = 20.0, GRIDN = 20)

super_slow_anneal=copy.deepcopy(slow_anneal)
super_slow_anneal[0][1]['anneal_sched']['start_temp'] = 64.0
super_slow_anneal[0][1]['anneal_sched']['iterations'] = 600

KERNEL_CONFIGS = {
                  'anneal_slow_400' : {'ITERS' : 400, 
                                       'kernels' : slow_anneal},
                  'anneal_slow_800' : {'ITERS' : 800, 
                                       'kernels' : super_slow_anneal},
    'debug_2' : {'ITERS' : 2, 
                                       'kernels' : slow_anneal},

                  }

pickle.dump(slow_anneal, open("kernel.config", 'w'))





@files(CELEGANS_DB, td("celegans_data.pickle"))
def data_preprocess(dbname, 
                    outfile):
    """
    """
    conn = sqlite3.connect(CELEGANS_DB)

    neurons = pandas.io.sql.read_frame("select * from Cells order by cell_id", 
                                 conn, index_col="cell_id" )


    canonical_neuron_ordering = np.array(neurons.index)
    NEURON_N = len(canonical_neuron_ordering)
    conn_matrix = np.zeros((NEURON_N, NEURON_N), 
                           dtype = [('chemical', np.int32), 
                                    ('electrical', np.int32)])

    neurons['index'] = np.arange(len(neurons), dtype=np.int)

    s = pandas.io.sql.read_frame("select * from Synapses", 
                                     conn)
    idx = neurons['index']
    print idx
    s['from_idx'] = np.array(idx[s['from_id']])
    s['to_idx'] = np.array(idx[s['to_id']])



    chem = s[s['synapse_type'] == 'C']
    elec = s[s['synapse_type'] == 'E']

    for si, s in chem.iterrows():
        conn_matrix['chemical'][s['from_idx'], s['to_idx']] += s['count']

    for si, s in elec.iterrows():
        conn_matrix['electrical'][s['from_idx'], s['to_idx']] += s['count']
        #conn_matrix['electrical'][s['to_idx'], s['from_idx']] = s['count']

    # check if electrical is symmetric
    np.testing.assert_almost_equal(conn_matrix['electrical'], conn_matrix['electrical'].T)

    pickle.dump({'canonical_neuron_ordering' : canonical_neuron_ordering, 
                 'conn_matrix' : conn_matrix, 
                 #'connections' : connections, 
                 'neurons' : neurons}, 
                open(outfile, 'w'), -1)





# @split(td('celegans_data.pickle'), [td(x) for x in ['celegans.both.data.pickle', 
#                                                     'celegans.electrical.data.pickle', 
#                                                     'celegans.chemical.data.pickle']])
# def data_celegans_adj(infile, (both_file, electrical_file, chemical_file)):
#     data = pickle.load(open(infile, 'r'))
#     conn_matrix = data['conn_matrix']
#     neurons = data['neurons']
#     canonical_neuron_ordering = data['canonical_neuron_ordering']
#     NEURON_N = len(canonical_neuron_ordering)
#     dist_matrix = np.zeros((NEURON_N, NEURON_N), 
#                            dtype=[('link', np.uint8), 
#                                   ('distance', np.float32)])
#     print neurons
#     # compute distance
#     for n1_i, n1 in enumerate(canonical_neuron_ordering):
#         for n2_i, n2 in enumerate(canonical_neuron_ordering):
            
#             dist_matrix[n1_i, n2_i]['distance'] = np.abs(neurons.irow(n1_i)['soma_pos'] - neurons.irow(n2_i)['soma_pos'])
    
#     adj_mat_chem = conn_matrix['chemical'] > 0
#     adj_mat_elec = conn_matrix['electrical'] > 0
#     adj_mat_both = np.logical_or(adj_mat_chem, adj_mat_elec)
    
#     dist_matrix['link'] = adj_mat_both
#     pickle.dump({'dist_matrix' : dist_matrix, 
#                  'infile' : infile}, open(both_file, 'w'))
    
#     dist_matrix['link'] = adj_mat_chem
#     pickle.dump({'dist_matrix' : dist_matrix, 
#                  'infile' : infile}, open(chemical_file, 'w'))
    
#     dist_matrix['link'] = adj_mat_elec
#     pickle.dump({'dist_matrix' : dist_matrix, 
#                  'infile' : infile}, open(electrical_file, 'w'))

def create_latents_2r_param():
    infile = td('celegans_data.pickle')
    # for bb_hpi in range(len(BB_HPS)):
    #     base = td('celegans.2r.bb.%02d' % bb_hpi)
    #     yield infile, [base + '.data', base+'.latent', base+'.meta'], 'BetaBernoulli', bb_hpi

    # for gp_hpi in range(len(GP_HPS)):
    #     base = td('celegans.2r.gp.%02d' % gp_hpi)
    #     yield infile, [base + '.data', base+'.latent', base+'.meta'], 'GammaPoisson', gp_hpi

    # for ld_hpi in range(len(LD_HPS)):
    #     base = td('celegans.2r.ld.%02d' % ld_hpi)
    #     yield infile, [base + '.data', base+'.latent', base+'.meta'], 'LogisticDistance', ld_hpi

    # for ldfl_hpi in range(len(LDFL_HPS)):
    #     base = td('celegans.2r.ldfl.%02d' % ldfl_hpi)
    #     yield infile, [base + '.data', base+'.latent', base+'.meta'], 'LogisticDistanceFixedLambda', ldfl_hpi

    # for sdb_hpi in range(len(SDB_HPS)):
    #     base = td('celegans.2r.sdb.%02d' % sdb_hpi)
    #     yield infile, [base + '.data', base+'.latent', base+'.meta'], 'SquareDistanceBump', sdb_hpi

    # for ndfw_hpi in range(len(NDFW_HPS)):
    #     base = td('celegans.2r.ndfw.%02d' % ndfw_hpi)
    #     yield infile, [base + '.data', base+'.latent', base+'.meta'], 'NormalDistanceFixedWidth', ndfw_hpi
        
    # for edp_hpi in range(len(EDP_HPS)):
    #     base = td('celegans.2r.edp.%02d' % edp_hpi)
    #     yield infile, [base + '.data', base+'.latent', base+'.meta'], 'ExponentialDistancePoisson', edp_hpi

    for ldp_hpi in range(len(LDP_HPS)):
        base = td('celegans.2r.ldp.%02d' % ldp_hpi)
        yield infile, [base + '.data', base+'.latent', base+'.meta'], 'LogisticDistancePoisson', ldp_hpi
        

@follows(data_preprocess)
@files(create_latents_2r_param)
def create_latents_2r_paramed(infile, 
                              (data_filename, latent_filename, meta_filename), 
                              model_name, hp_i):


    data = pickle.load(open(infile, 'r'))
    conn_matrix = data['conn_matrix']
    neurons = data['neurons']
    canonical_neuron_ordering = data['canonical_neuron_ordering']
    NEURON_N = len(canonical_neuron_ordering)
    dist_matrix = np.zeros((NEURON_N, NEURON_N), dtype=np.float32)


    for n1_i, n1 in enumerate(canonical_neuron_ordering):
        for n2_i, n2 in enumerate(canonical_neuron_ordering):
            dist_matrix[n1_i, n2_i] = np.abs(neurons.loc[n1]['soma_pos'] - neurons.loc[n2]['soma_pos'])

    if model_name == "BetaBernoulli":
        chem_conn = conn_matrix['chemical'] > 0 

        elec_conn = conn_matrix['electrical'] > 0 

        irm_latent, irm_data = irm.irmio.default_graph_init(elec_conn, model_name, 
                                                            extra_conn=[chem_conn])

        HPS = {'alpha' : BB_HPS[hp_i][0], 
               'beta' : BB_HPS[hp_i][1]}
 
    elif model_name == "LogisticDistance":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        chem_conn['link'] =  conn_matrix['chemical'] > 0 
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] > 0 
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'mu_hp' : LD_HPS[hp_i][0], 
               'lambda_hp' : LD_HPS[hp_i][1], 
               'p_max' : LD_HPS[hp_i][2], 
               'p_min' : LD_HPS[hp_i][3]}

    elif model_name == "LogisticDistanceFixedLambda":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        chem_conn['link'] =  conn_matrix['chemical'] > 0 
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] > 0 
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'mu_hp' : LDFL_HPS[hp_i][0], 
               'lambda' : LDFL_HPS[hp_i][1], 
               'p_min' : LDFL_HPS[hp_i][2],
               'p_scale_alpha_hp' : LDFL_HPS[hp_i][3], 
               'p_scale_beta_hp' : LDFL_HPS[hp_i][4]}

    elif model_name == "NormalDistanceFixedWidth":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        chem_conn['link'] =  conn_matrix['chemical'] > 0 
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] > 0 
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'mu_hp' : NDFW_HPS[hp_i][0], 
               'p_min' : NDFW_HPS[hp_i][1], 
               'width' : NDFW_HPS[hp_i][2], 
               'p_alpha' : NDFW_HPS[hp_i][3], 
               'p_beta' : NDFW_HPS[hp_i][4]}

    elif model_name == "SquareDistanceBump":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        chem_conn['link'] =  conn_matrix['chemical'] > 0 
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] > 0 
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'p_alpha' : SDB_HPS[hp_i][0], 
               'p_beta' : SDB_HPS[hp_i][1], 
               'mu_hp' : SDB_HPS[hp_i][2], 
               'p_min' : SDB_HPS[hp_i][3], 
               'param_weight' : SDB_HPS[hp_i][4], 
               'param_max_distance' : SDB_HPS[hp_i][5]}

    elif model_name == "GammaPoisson":
        chem_conn = conn_matrix['chemical'].astype(np.uint32)

        elec_conn = conn_matrix['electrical'].astype(np.uint32)

        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'alpha' : GP_HPS[hp_i][0], 
               'beta' : GP_HPS[hp_i][1]}
               
    elif model_name == "ExponentialDistancePoisson":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                             dtype=[('link', np.int32), 
                                    ('distance', np.float32)])

        scale = EDP_HPS[hp_i][2]
        chem_conn['link'] =  conn_matrix['chemical'] * scale
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                             dtype=[('link', np.int32), 
                                    ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] * scale
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'mu_hp' : EDP_HPS[hp_i][0], 
               'rate_scale_hp' : EDP_HPS[hp_i][1]}

    elif model_name == "LogisticDistancePoisson":
        # compute distance
                
        chem_conn = np.zeros((NEURON_N, NEURON_N), 
                             dtype=[('link', np.int32), 
                                    ('distance', np.float32)])

        scale = LDP_HPS[hp_i][4]
        chem_conn['link'] =  conn_matrix['chemical'] * scale
        chem_conn['distance'] = dist_matrix

        elec_conn = np.zeros((NEURON_N, NEURON_N), 
                             dtype=[('link', np.int32), 
                                    ('distance', np.float32)])

        elec_conn['link'] =  conn_matrix['electrical'] * scale
        elec_conn['distance'] = dist_matrix


        irm_latent, irm_data = irm.irmio.default_graph_init(chem_conn, model_name, 
                                                            extra_conn=[elec_conn])

        HPS = {'mu_hp' : LDP_HPS[hp_i][0], 
               'lambda' : LDP_HPS[hp_i][1], 
               'rate_min' : LDP_HPS[hp_i][2],
               'rate_scale_hp' : LDP_HPS[hp_i][3], 
           }

   
    irm_latent['relations']['R1']['hps'] = HPS
    irm_latent['relations']['R2']['hps'] = HPS

    pickle.dump(irm_latent, open(latent_filename, 'w'))
    pickle.dump(irm_data, open(data_filename, 'w'))
    pickle.dump({'infile' : infile, 
                 'r1' : chem_conn, 
                 'r2' : elec_conn},
                open(meta_filename, 'w'))


    
# @transform(data_create_thold, suffix(".pickle"), ".png")
# def plot_data_raw((infile, ), outfile):
#     x = pickle.load(open(infile, 'r'))
#     cells = x['cells']
#     print cells
#     f = pylab.figure()
#     ax  = f.add_subplot(1, 1, 1)
#     ai = np.argsort(cells['y'])[::-1]
#     cm = x['conn_mat']
#     cm = cm[ai]
#     cm = cm[:, ai]

#     ax.imshow(cm, interpolation='nearest', cmap=pylab.cm.Greys)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     f.tight_layout()
#     f.savefig(outfile, dpi=200)
    


# def create_latents_srm_params():
#     for a in create_tholds():
#         inf = a[1][0]
#         for mli, mulamb in enumerate(MULAMBS):
#             for pi, p in enumerate(PMAXS):
#                 for distvars in ['x', 'xyz', 'yz']:
#                     outf_base = inf[:-len('.data.pickle')]
#                     outf = "%s.ld.%d.%d.%s" % (outf_base, mli, pi, distvars)
#                     yield inf, [outf + '.data', 
#                                 outf + '.latent', outf + '.meta'], distvars, mulamb, 0.01, p


# @follows(data_create_thold)
# @files(create_latents_srm_params)
# def create_latents_srm(infile, 
#                        (data_filename, latent_filename, meta_filename), 
#                        distvars, mulamb, p_min, p_max):


#     irm_latent, irm_data = models.create_conn_dist(infile, distvars)

#     HPS = {'mu_hp' : mulamb,
#            'lambda_hp' : mulamb,
#            'p_min' : p_min, 
#            'p_max' : p_max}

#     irm_latent['relations']['R1']['hps'] = HPS

#     pickle.dump(irm_latent, open(latent_filename, 'w'))
#     pickle.dump(irm_data, open(data_filename, 'w'))
#     pickle.dump({'infile' : infile, 
#                  }, open(meta_filename, 'w'))




# def create_latents_bb_params():
#     for a in create_tholds():
#         inf = a[1][0]
#         outf_base = inf[:-len('.data.pickle')]
#         outf = "%s.bb" % (outf_base)
#         yield inf, [outf + '.data', 
#                     outf + '.latent', outf + '.meta']


# @follows(data_create_thold)
# @files(create_latents_bb_params)
# def create_latents_bb(infile, 
#                        (data_filename, latent_filename, meta_filename)):


#     d = pickle.load(open(infile, 'r'))
#     conn_mat = d['conn_mat']
#     model_name = "BetaBernoulliNonConj"
#     irm_latent, irm_data = irm.irmio.default_graph_init(conn_mat, model_name)

#     HPS = {'alpha' : BB_ALPHAS[0], 
#            'beta' : BB_BETAS[0]}

#     irm_latent['relations']['R1']['hps'] = HPS

#     pickle.dump(irm_latent, open(latent_filename, 'w'))
#     pickle.dump(irm_data, open(data_filename, 'w'))
#     pickle.dump({'infile' : infile, 
#                  }, open(meta_filename, 'w'))


# def create_latents_xsoma_params():
#     for a in create_tholds():
#         inf = a[1][0]

#         outf = td("retina.xsoma")
#         yield inf, [outf + '.data', 
#                     outf + '.latent', outf + '.meta']
#         return # only one of these


# @follows(data_create_thold)
# @files(create_latents_xsoma_params)
# def create_latents_xsoma(infile, 
#                          (data_filename, latent_filename, meta_filename)):
#     """
#     Just the soma x-positions
#     """
#     d = pickle.load(open(infile, 'r'))
#     cells =  d['cells']


#     feature_desc = {'soma_x' : {'data' : to_f32(cells['x']), 
#                                 'model' : 'NormalInverseChiSq'}, 
#     }

#     # FIXME do we have to set the initial hypers? 

#     latent, data = connattribio.create_mm(feature_desc)
#     HPS = {'kappa' : 0.0001, 
#            'mu' : 50.0, 
#            'sigmasq' : 0.1, 
#            'nu' : 15.0}

#     latent['relations']['r_soma_x']['hps'] = HPS

#     pickle.dump(latent, open(latent_filename, 'w'))
#     pickle.dump(data, open(data_filename, 'w'))
#     pickle.dump({'infile' : infile, 
#                  }, open(meta_filename, 'w'))



# def create_latents_clist_params():
#     for a in create_tholds():
#         inf = a[1][0]
#         for vsi, vs in enumerate(VAR_SCALES):
#             for cki, comp_k in enumerate(COMP_KS):

#                 outf_base = inf[:-len('.data.pickle')]
#                 outf = "%s.clist.%d.%d" % (outf_base, vsi, comp_k)
#                 yield inf, [outf + '.data', 
#                             outf + '.latent', outf + '.meta'], vs, comp_k


# @follows(data_create_thold)
# @files(create_latents_clist_params)
# def create_latents_clist(infile, 
#                          (data_filename, latent_filename, meta_filename), 
#                          vs, COMP_K):

#     d = pickle.load(open(infile, 'r'))
#     cells =  d['cells']

#     print "Creating mixmodel data", vs, COMP_K
#     contact_x_list = models.create_mixmodeldata(cells['contact_x_list'], 
#                                                 20, 120)
#     print "done"
#     feature_desc = {
#         'contact_x_list' : {'data' : contact_x_list,
#                             'model' : 'MixtureModelDistribution'}

#     }

#     latent, data = connattribio.create_mm(feature_desc)
#     latent['relations']['r_contact_x_list']['hps'] = {'comp_k': COMP_K, 
#                                                       'var_scale' : vs, 
#                                                       'dir_alpha' : 1.0}

#     pickle.dump(latent, open(latent_filename, 'w'))
#     pickle.dump(data, open(data_filename, 'w'))
#     pickle.dump({'infile' : infile, 
#                  }, open(meta_filename, 'w'))

# def create_latents_srm_clist_xsoma_params():
#     for a in create_tholds():
#         inf = a[1][0]
#         outf_base = inf[:-len('.data.pickle')]
#         for ml_i in range(len(MULAMBS)):
#             for pmax_i in range(len(PMAXS)):
#                 for vars in ['xyz']: # FXIME only doing one of these right now
#                     for var_scale in range(len(VAR_SCALES)):
#                         for comp_k in COMP_KS:

#                             outf = '%s.srm_clist_xsoma.%d.%d.%s.%d.%d' % (outf_base, 
#                                                                  ml_i, pmax_i, vars, 
#                                                                  var_scale, comp_k)

#                             yield inf, [outf + '.data', 
#                                         outf + '.latent', outf + '.meta'], vars, MULAMBS[ml_i], PMAXS[pmax_i], VAR_SCALES[var_scale], comp_k


# @follows(data_create_thold)
# @files(create_latents_srm_clist_xsoma_params)
# def create_latents_srm_clist_xsoma(infile, 
#                                   (data_filename, latent_filename, meta_filename), 
#                                   distvars, mulamb, p_max, 
#                                   var_scale, COMP_K):

#     d = pickle.load(open(infile, 'r'))
#     cells =  d['cells']
    
#     graph_latent, graph_data = models.create_conn_dist(infile, distvars)
#     assert mulamb > 0 
#     HPS = {'mu_hp' : mulamb,
#            'lambda_hp' : mulamb,
#            'p_min' : 0.01, 
#            'p_max' : p_max}

#     graph_latent['relations']['R1']['hps'] = HPS



#     contact_x_list = models.create_mixmodeldata(cells['contact_x_list'], 
#                                                 20, 120)
#     feature_desc = {
#         'contact_x_list' : {'data' : contact_x_list,
#                             'model' : 'MixtureModelDistribution'}, 
#         'soma_x' : {'data' : to_f32(cells['x']), 
#                                 'model' : 'NormalInverseChiSq'}, 
#     }


#     feature_latent, feature_data = connattribio.create_mm(feature_desc)
#     feature_latent['relations']['r_contact_x_list']['hps'] = {'comp_k': COMP_K, 
#                                                               'var_scale' : var_scale, 
#                                                               'dir_alpha' : 1.0}

#     soma_x_HPS = {'kappa' : 0.0001, 
#                   'mu' : 50.0, 
#                   'sigmasq' : 0.1, 
#                   'nu' : 15.0}

#     graph_latent['relations']['R1']['hps'] = HPS

#     feature_latent['relations']['r_soma_x']['hps'] = soma_x_HPS

#     latent, data = models.merge_graph_features(graph_latent, graph_data, 
#                                 feature_latent, feature_data, 
#                                 'd1')


#     pickle.dump(latent, open(latent_filename, 'w'))
#     pickle.dump(data, open(data_filename, 'w'))
#     pickle.dump({'infile' : infile, 
#                  }, open(meta_filename, 'w'))


def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))


# # def init_generator():
# #     for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
# #         for data_filename in get_dataset("%s-*%s*" % (data_name, cv_config_name)):
# #             name, _ = os.path.splitext(data_filename)

# #             # now generate one of these for every single cv dataset:
# #             yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]

# # def cv_generator():
# #     for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
# #         for data_filename in get_dataset(data_name):
# #             name, _ = os.path.splitext(data_filename)
# #             for cv_i in range(CV_CONFIGS[cv_config_name]['N']):
# #                 cv_name_base = "%s-%s.%02d.cv" % (name, cv_config_name, cv_i)
                
# #                 yield data_filename, [cv_name_base + ".data",
# #                                       cv_name_base + ".latent",
# #                                       cv_name_base + ".meta"], cv_i, cv_config_name, CV_CONFIGS[cv_config_name]



# # @follows(create_latents_srm)
# # @follows(create_latents_clist)
# # @follows(create_latents_xsoma)
# # @follows(create_latents_bb)
# # @follows(create_latents_srm_clist_xsoma)
# # @files(cv_generator)
# # def create_cv(data_filename, (out_data_filename,
# #                               out_latent_filename,
# #                               out_meta_filename), cv_i, cv_config_name, cv_config):
# #     """ 
# #     Creates a single cross-validated data set 
# #     """
# #     basename, _ = os.path.splitext(data_filename)
# #     latent_filename = basename + ".latent"
# #     meta_filename = basename + ".meta"


# #     data = pickle.load(open(data_filename))
# #     shape = data['relations']['R1']['data'].shape
# #     N =  shape[0] * shape[1]
# #     if cv_config['type'] == 'nfold':
# #         np.random.seed(0) # set the seed

# #         perm = np.random.permutation(N)
# #         subset_size = N / cv_config['N']
# #         subset = perm[cv_i * subset_size:(cv_i+1)*subset_size]
        
# #         observed = np.ones(N, dtype=np.uint8)
# #         observed[subset] = 0
# #         data['relations']['R1']['observed'] = np.reshape(observed, shape)

# #     else:
# #         raise Exception("Unknown cv type")
    
# #     pickle.dump(data, open(out_data_filename, 'w'))
                    
# #     latent = pickle.load(open(latent_filename))
# #     pickle.dump(latent, open(out_latent_filename, 'w'))
                    
# #     meta = pickle.load(open(meta_filename))
# #     meta['cv'] = {'cv_i' : cv_i,
# #                   'cv_config_name' : cv_config_name}
    
# #     pickle.dump(meta, open(out_meta_filename, 'w'))

            
# # @follows(create_latents_srm)
# # @follows(create_latents_clist)
# # @follows(create_latents_xsoma)
# # @follows(create_latents_bb)
# # @follows(create_latents_srm_clist_xsoma)
# # @follows(create_cv)
# # @files(init_generator)
# # def create_inits(data_filename, out_filenames, init_config_name, init_config):
# #     basename, _ = os.path.splitext(data_filename)
# #     latent_filename = basename + ".latent"
    
# #     irm.experiments.create_init(latent_filename, data_filename, 
# #                                 out_filenames, 
# #                                 init= init_config['config'], 
# #                                 keep_ground_truth=False)



# # def experiment_generator():
# #     for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
# #         for data_filename in get_dataset("%s-*%s*" % (data_name, cv_config_name)):
# #             name, _ = os.path.splitext(data_filename)

# #             inits = ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
# #             exp_name = "%s-%s-%s.wait" % (data_filename, init_config_name, kernel_config_name)
# #             yield [data_filename, inits], exp_name, kernel_config_name

# # @follows(create_volume)
# # @follows(create_inits)
# # @files(experiment_generator)
# # def run_exp((data_filename, inits), wait_file, kernel_config_name):
# #     # put the filenames in the data
# #     print "Putting the file", data_filename, "in the bucket" 

# #     irm.experiments.to_bucket(data_filename, VOLUME_NAME)

# #     [irm.experiments.to_bucket(init_f, VOLUME_NAME) for init_f in inits]
# #     kernel_config_filename = kernel_config_name + ".pickle"

# #     kc = KERNEL_CONFIGS[kernel_config_name]
# #     ITERS = kc['ITERS']
# #     kernel_config = kc['kernels']
# #     fixed_k = kc.get('fixed_k', False)
# #     cores = kc.get('cores', DEFAULT_CORES)
# #     relation_class = kc.get('relation_class', DEFAULT_RELATION)

# #     pickle.dump(kernel_config, open(kernel_config_filename, 'w'))

# #     irm.experiments.to_bucket(kernel_config_filename, VOLUME_NAME)


# #     CHAINS_TO_RUN = len(inits)

    
# #     jids = []

# #     for init_i, init in enumerate(inits):
# #         jid = multyvac.submit(irm.experiments.inference_run, 
# #                               init, 
# #                               data_filename, 
# #                               kernel_config_filename, 
# #                               ITERS, 
# #                               init_i, 
# #                               VOLUME_NAME, 
# #                               None, 
# #                               fixed_k, 
# #                               relation_class = relation_class, 
# #                               cores = cores, 
# #                               _name="%s-%s-%s" % (data_filename, init, 
# #                                                   kernel_config_name), 
# #                               _layer = MULTYVAC_LAYER,
# #                               _multicore = cores, 
# #                               _core = 'f2')
# #         jids.append(jid)


# #     pickle.dump({'jids' : jids, 
# #                 'data_filename' : data_filename, 
# #                 'inits' : inits, 
# #                 'kernel_config_name' : kernel_config_name}, 
# #                 open(wait_file, 'w'))


# # @transform(run_exp, suffix('.wait'), '.samples')
# # def get_results(exp_wait, exp_results):
# #     d = pickle.load(open(exp_wait, 'r'))
    
# #     chains = []
# #     # reorg on a per-seed basis
# #     error_count = 0
# #     for jid in d['jids']:
# #         job = multyvac.get(jid)
# #         print "waiting on", jid
# #         job.wait()
# #         if job.status  != "done" :
# #             print "ERROR", jid, job.status, exp_wait
# #             fid = open("error.%s" % jid, 'w')
# #             fid.write("error! %s\n" % exp_wait)
# #             fid.close()
# #             error_count += 1
# #         else:
# #             print "downloading", jid, "results"
# #             chain_data = job.get_result()

# #             chains.append({'scores' : chain_data[0], 
# #                            'state' : chain_data[1], 
# #                            'times' : chain_data[2], 
# #                            'latents' : chain_data[3]})
# #     if error_count > 0.75 * len(d['jids']):
# #         print "MOSTLY FAIL", exp_wait
        
# #     pickle.dump({'chains' : chains, 
# #                  'exp' : d}, 
# #                 open(exp_results, 'w'))

def experiment_generator(EXPERIMENTS, CV_CONFIGS, INIT_CONFIGS, get_dataset, td):
    for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
        data_filename = td(data_name +".data")

        df = "%s-%s-%s-%s" % (data_name, cv_config_name, init_config_name, kernel_config_name)
        
        out_files = [td(df + x) for x in [ ".samples",
                                       ".cvdata", ".inits"]]
        init_config = INIT_CONFIGS[init_config_name]
        cv_config = CV_CONFIGS[cv_config_name]
        
        yield data_filename, out_files, cv_config_name, init_config_name, kernel_config_name, init_config, cv_config


# @follows(create_latents_srm)
# @follows(create_latents_clist)
# @follows(create_latents_xsoma)
# @follows(create_latents_bb)
# @follows(create_latents_srm_clist_xsoma)
@follows(create_latents_2r_param)
@jobs_limit(1)
@files(list(experiment_generator(EXPERIMENTS, CV_CONFIGS,
                                 INIT_CONFIGS, get_dataset, td)))
def spark_run_experiments(data_filename, (out_samples, out_cv_data, out_inits), 
                          cv_config_name, init_config_name, kernel_config_name,
                          init_config, cv_config):

    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"
    meta_filename = basename + ".meta"
    

    data = pickle.load(open(data_filename))
    true_latent = pickle.load(open(latent_filename))
    meta = pickle.load(open(meta_filename))
    
    sc = SparkContext(batchSize=1)
    kc = KERNEL_CONFIGS[kernel_config_name]

    CV_N = cv_config['N']

    INIT_N = init_config['N']
    print "WE ARE RUNNING THIS THING" 
    def cv_create(cv_i):
        """ 
        returns key, (data, meta)
        """
        cv_key = "%s.%d" % (cv_config_name, cv_i)
        return cv_key, cv.create_cv_pure(data, #_broadcast.value, 
                                      meta, cv_i,
                                      cv_config_name, cv_config)

    
    cv_data_rdd = sc.parallelize(range(CV_N), 
                                 CV_N).map(cv_create).cache()
    
    def create_init_flat((key, (data, meta))):
        """
        returns latent
        """
        res = []
        for latent_i, latent in enumerate( cv.create_init_pure(true_latent, data, INIT_N,
                                                            init_config['config'])):
            yield "%s.%s" % (key, latent_i), (data, meta, latent)


    init_latents_rdd  = cv_data_rdd.flatMap(create_init_flat)

    def inference((data, meta, init)):

        return cv.run_exp_pure(data, init, kernel_config_name, 0, kc)
       # FIXME do we care about this seed? 
        
    joined = init_latents_rdd.repartition(CV_N*INIT_N).cache()
    print "THERE ARE", joined.getNumPartitions(), "partitions" 
    results = joined.mapValues(inference)


    for rdd, name in [(results, out_samples),
                      (init_latents_rdd, out_inits),
                      (cv_data_rdd, out_cv_data)]:
        url = sparkutil.util.s3n_url(S3_BUCKET, S3_PATH, name)
        sparkutil.util.s3n_delete(url)
        rdd.saveAsPickleFile(url)
        pickle.dump({'url' : url}, open(name, 'w'))
    
    sc.stop()

    

@transform(spark_run_experiments, suffix('.samples'), '.samples.pickle')
def get_samples((exp_samples, exp_cvdata, exp_inits), out_filename):
    sample_metadata = pickle.load(open(exp_samples, 'r'))
    
    sc = SparkContext()
    results_rdd = sc.pickleFile(sample_metadata['url'])

    sparkutil.util.save_rdd_elements(results_rdd, out_filename, S3_BUCKET, S3_PATH)
    
    sc.stop()

    
@transform(spark_run_experiments, suffix('.samples'), '.cvdata.pickle')
def get_cvdata((exp_samples, exp_cvdata, exp_inits), out_filename):
    cvdata_metadata = pickle.load(open(exp_cvdata, 'r'))
    
    sc = SparkContext()
    results_rdd = sc.pickleFile(cvdata_metadata['url'])
    pickle.dump(results_rdd.collect(),
                open(out_filename, 'w'))
    sc.stop()






@follows(get_samples)
@transform(get_samples, suffix(".samples.pickle"), ".samples.organized.sentinel")
def samples_organize(infile, outfile):
    associated_files_list = pickle.load(open(infile, 'r'))
    print "="*80
    
    print "infile=", infile
    print associated_files_list
    for f in associated_files_list:
        print "\n"
        a = pickle.load(open(f, 'r'))
        key_base, cv_id, samp_id = a[0].split('.')
        dir_name = os.path.join(outfile[:-9], cv_id)
        print "MAKING", dir_name, "cv_id=", cv_id, "samp_id=", samp_id
        try:
            os.makedirs(dir_name)
        except OSError:
            pass
        
        filename = os.path.join(dir_name, '%s.pickle' % samp_id)
        source_path = os.path.abspath(f)
        assert os.path.exists(source_path)
        print "Linking", source_path, filename
        try:
            os.remove(filename)
        except OSError:
            pass
        os.symlink(source_path, filename)
    fid = open(outfile, 'w')
    fid.write("")
    fid.close()

@follows(get_samples)
@follows(get_cvdata)
@follows(samples_organize)
@subdivide(get_cvdata, formatter(".+/(?P<base>.*).cvdata.pickle"), 
           "{path[0]}/{base[0]}.samples.organized/*/cv.data",
           # Output parameter: Glob matches any number of output file names
            "{path[0]}/{base[0]}.samples.organized")
# Extra parameter:  Append to this for output file names
def cvdata_organize(input_file, output_files, output_file_name_root):
    print "input_file=", input_file
    a = pickle.load(open(input_file, 'r'))
    for di, d in enumerate(a):
        key = d[0]
        a = key.split('.')[1]
        data, meta = d[1]
        pickle.dump(data, open(os.path.join(output_file_name_root, a, "cv.data"), 'w'))
        pickle.dump(meta, open(os.path.join(output_file_name_root, a, "cv.meta"), 'w'))

PRED_EVALS= np.logspace(-4, 0, 41) # np.linspace(0, 1.0, 41)

@follows(samples_organize)
@follows(cvdata_organize)
@collate("sparkdatacv/*.samples.organized/*",
         regex(r"(sparkdatacv/.+.samples.organized)/(\d+)"),
         [r"\1/predlinks.pickle", r'\1/assign.pickle'])
def cv_collate_predlinks_assign(cv_dirs, (predlinks_outfile,
                                          assign_outfile)):
    """
    aggregate across samples and cross-validations

    # FIXME FIXME mult-link prediction? How to handle? 

    """
    print "THIS IS A RUN", predlinks_outfile
    for d in cv_dirs:
        print "cv dir", d

    
    base = os.path.dirname(cv_dirs[0])[:-len('.samples.organized')]
    s = base.split('-')

    input_basename = s[0]
    input_data = pickle.load(open(input_basename + ".data"))
    input_latent = pickle.load(open(input_basename + ".latent"))
    for relation_name in input_data['relations']:
        data_conn = input_data['relations'][relation_name]['data']
        model_name= input_data['relations'][relation_name]['model']

        input_meta = pickle.load(open(input_basename + ".meta"))['infile']
        meta = pickle.load(open(input_meta, 'r'))

        print "META IS", meta.keys()
        print meta['neurons'].dtypes

        N = len(data_conn)

        true_assign = meta['neurons']['cell_class'] # input_latent['domains']['d1']['assignment']

        print "for", input_basename, "there are", len(np.unique(true_assign)), "classes"
        if model_name == "BetaBernoulliNonConj":
            truth_mat = data_conn
        elif model_name == "LogisticDistance":
            truth_mat = data_conn['link']
        elif model_name == "LogisticDistancePoisson":
            # For count data we just binarize with > 0 

            truth_mat = data_conn['link'] > 0 

        predlinks_outputs = []
        assignments_outputs = []


        N = len(data_conn)

        for cv_dir in cv_dirs:
            cv_data = pickle.load(open(os.path.join(cv_dir, 'cv.data'), 'r'))
            # get the cv idx for later use
            cv_idx = int(os.path.basename(cv_dir))




            # FIGURE OUT WHICH ENTRIES WERE MISSING
            heldout_idx = np.argwhere((cv_data['relations'][relation_name]['observed'].flatten() == 0)).flatten()
            heldout_true_vals = truth_mat.flatten()[heldout_idx]

            heldout_true_vals_t_idx = np.argwhere(heldout_true_vals > 0).flatten()
            heldout_true_vals_f_idx = np.argwhere(heldout_true_vals == 0).flatten()

            sample_file_str =os.path.join(cv_dir, r"[0-9]*.pickle")
            print "files are"
            for sample_name in glob.glob(sample_file_str):
                chain_i = int(os.path.basename(sample_name)[:-(len('.pickle'))])
                print chain_i

                sample = pickle.load(open(sample_name, 'r'))
                inf_results = sample[1]['res'] # state
                irm_latent_samp = inf_results[1]
                scores = inf_results[0]
                print irm_latent_samp.keys()
                # compute full prediction matrix 
                pred = predutil.compute_prob_matrix(irm_latent_samp, input_data, 
                                                    model_name)
                pf_heldout = pred.flatten()[heldout_idx]

                for pred_thold  in PRED_EVALS:
                    pm = pf_heldout[heldout_true_vals_t_idx]
                    t_t = np.sum(pm >=  pred_thold)
                    t_f = np.sum(pm <= pred_thold)
                    pm = pf_heldout[heldout_true_vals_f_idx]
                    f_t = np.sum(pm >= pred_thold)
                    f_f = np.sum(pm <= pred_thold)
                    predlinks_outputs.append({'chain_i' : chain_i, 
                                              'score' : scores[-1], 
                                              'pred_thold' : pred_thold,
                                              'cv_idx' : cv_idx, 
                                              't_t' : t_t, 
                                              't_f' : t_f, 
                                              'f_t' : f_t, 
                                              'f_f' : f_f, 
                                              't_tot' : len(heldout_true_vals_t_idx), 
                                              'f_tot' : len(heldout_true_vals_f_idx), 
                                              'relation' : relation_name
                                          })
                assignments_outputs.append({'chain_i' : chain_i,
                                            'score' : scores[-1],
                                            'cv_idx' : cv_idx, 
                                            'true_assign' : true_assign,
                                            'assign' : irm_latent_samp['domains']['d1']['assignment'], 
                                            'relation' : relation_name})
                                            


    predlinks_df = pandas.DataFrame(predlinks_outputs)
    pickle.dump({'df' : predlinks_df}, 
                 open(predlinks_outfile, 'w'))

    a_df = pandas.DataFrame(assignments_outputs)
    pickle.dump({'df' : a_df}, 
                 open(assign_outfile, 'w'))

                  
    
# # ## TODO GET OLD PLOTTING FROM PROCESS.PY
# # @transform(get_results, suffix(".samples"), 
# #            ".latent.pdf")
# # def plot_best_cluster_latent(exp_results, 
# #                      out_filename):
    
# #     sample_d = pickle.load(open(exp_results))
# #     chains = sample_d['chains']
    
# #     exp = sample_d['exp']
# #     data_filename = exp['data_filename']
# #     data = pickle.load(open(data_filename))
# #     data_basename, _ = os.path.splitext(data_filename)
# #     meta = pickle.load(open(data_basename + ".meta"))

# #     meta_infile = meta['infile']
# #     print "meta_infile=", meta_infile

# #     d = pickle.load(open(meta_infile, 'r'))
# #     conn = d['conn_mat']
# #     cells = d['cells']

# #     cell_types = cells['type_id']

# #     chains = [c for c in chains if type(c['scores']) != int]
# #     CHAINN = len(chains)

# #     chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
# #     chain_pos = 0

# #     best_chain_i = chains_sorted_order[chain_pos]
# #     best_chain = chains[best_chain_i]
# #     sample_latent = best_chain['state']
# #     cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])
# #     if 'R1' in data['relations']:
# #         irm.experiments.plot_latent(sample_latent, 
# #                                     data['relations']['R1']['data'], 
# #                                     out_filename, 
# #                                     model=data['relations']['R1']['model'])
# #     else:
# #         # dont do the clist ones
# #         file(out_filename, 'w').write("test")

# # @transform(get_results, suffix(".samples"), 
# #            ".z.pdf")
# # def plot_z_matrix(exp_results, 
# #                   out_filename):
# #     # debug
# #     # fid = open(out_filename, 'w')
# #     # fid.write("test")
# #     # fid.close()
# #     # return

# #     sample_d = pickle.load(open(exp_results))
# #     chains = sample_d['chains']
# #     exp = sample_d['exp']
# #     print "exp=", exp
# #     data_filename = exp['data_filename']
# #     data = pickle.load(open(data_filename))
# #     data_basename, _ = os.path.splitext(data_filename)
# #     meta = pickle.load(open(data_basename + ".meta"))

# #     meta_infile = meta['infile']

# #     d = pickle.load(open(meta_infile, 'r'))
# #     conn = d['conn_mat']
# #     cells = d['cells']

# #     cell_types = cells['type_id']

# #     chains = [c for c in chains if type(c['scores']) != int]
# #     CHAINN = len(chains)

# #     # compute the z matrix 
# #     z_mat = np.zeros((len(cells), len(cells)))
# #     for ci, c in enumerate(chains):
# #         sample_latent = c['state']
# #         cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])
# #         ca = irm.util.canonicalize_assignment(cell_assignment)
# #         for u in np.unique(ca):
# #             ent_i = np.argwhere(ca == u).flatten()
# #             for ci in ent_i:
# #                 for cj in ent_i:
# #                     z_mat[ci, cj] += 1

# #     import scipy.cluster
# #     l = scipy.cluster.hierarchy.linkage(z_mat, method='ward')
    
    
# #     ca = np.array(scipy.cluster.hierarchy.leaves_list(l))
# #     z_mat = z_mat[ca]
# #     z_mat = z_mat[:, ca]


# #     import matplotlib.gridspec as grd
# #     f = pylab.figure(figsize=(8, 6))
# #     gs = grd.GridSpec(1, 2, width_ratios=[2,10 ], wspace=0.02)
# #     ax = f.add_subplot(gs[:, 1])
# #     im = ax.imshow(z_mat/CHAINN, cmap=pylab.cm.Greys, interpolation='nearest')
# #     ax.set_xticks([])
# #     ax.set_yticks([])
# #     ax.set_title("Cell coassignment probability") 
# #     ax.set_xlabel("cells")


# #     cbar_ax = f.add_axes([0.30, 0.15, 0.1, 0.02])
# #     f.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[0.0, 1.0])


# #     typeax = f.add_subplot(gs[:, 0])
# #     cells['coarse'][cells['type_id'] == 58] = 'bc'
# #     cmap = {None : 4, 
# #             'other': 4, 
# #             'nac' : 2, 
# #             'bc' : 1, 
# #             'mwac' : 3, 
# #             'gc' : 0}
# #     color_idx = [cmap[o] for o in cells['coarse']]        
# #     import brewer2mpl
# #     cmap = brewer2mpl.get_map('Set1', 'qualitative', 5).mpl_colormap
# #     typeax.scatter(cell_types, np.argsort(ca), edgecolor='none', 
# #                s=3, 
# #                c=color_idx, 
# #                alpha=0.8,
# #                cmap = cmap)
# #     p = np.array([0, 12, 24, 57, 72, 80])
# #     typeax.set_xticks(p)
# #     typeax.set_xticks(p[1:] - np.diff(p)/2. , minor=True)
# #     typeax.set_yticks([])
# #     typeax.set_xticklabels(['gc', 'bc', 'nac', 'mwac', 'other'], minor=True,
# #                            fontsize=8, rotation=90)
# #     typeax.set_xticklabels([])

# #     typeax.set_title("anatomist type (0-72)", fontsize=8)
# #     typeax.set_ylabel("cells")
# #     typeax.grid()

# #     # create colors
    
# #     typeax.set_xlim(0, 80)
# #     typeax.set_ylim(950, 0)


# #     # con = sqlite3.connect(RETINA_DB)
# #     # MAX_CONTACT_AREA=5.0
# #     # area_thold_min = 0.1
    
# #     # contacts_df = pandas.io.sql.read_frame("select * from contacts where area < %f and area > %f" % (MAX_CONTACT_AREA, area_thold_min), 
# #     #                                        con, index_col='id')

# #     # contacts_df.head()


# #     # for i in range(4):
# #     #     # pick points 
# #     #     which_row = np.argsort(ca).flatten()
# #     #     spos = [270, 580, 790, 930][i]
        
# #     #     #spos = np.argsort(ca).flatten()[spos]
# #     #     cell_row = np.argwhere(which_row == spos)[0]
# #     #     # np.argsort(ca).flatten()[spos]

# #     #     cell_row = cells.irow(cell_row)
# #     #     print cell_row
# #     #     cell_id = cell_row.index.values[0]
# #     #     print "cell_id=", cell_id
# #     #     typeax.scatter([cell_row['type_id']], 
# #     #                    [spos], c='k', s=20, edgecolor='k', 
# #     #                    facecolor='none')

# #     #     ax = f.add_subplot(gs[i, 0])
# #     #     c = contacts_df[contacts_df['from_id'] ==cell_id]
# #     #     ax.scatter(c['y'], c['x'], edgecolor='none', s=1, alpha=0.5, c='k')

# #     #     ax.scatter(cell_row['y'], cell_row['x'], s=40, c='r', edgecolor='none')


# #     #     ax.set_xlim(0, 120)
# #     #     ax.set_ylim(130, 50)
# #     #     ax.set_xticks([])
# #     #     ax.set_yticks([])
# #     # print cells.head()
# #     f.savefig(out_filename)

# # @transform(get_results, suffix(".samples"), [".hypers.pdf"])
# # def plot_hypers_vs_iter(exp_results, (plot_hypers_filename,)):

# #     sample_d = pickle.load(open(exp_results))
# #     chains = sample_d['chains']
    
# #     exp = sample_d['exp']
# #     data_filename = exp['data_filename']
# #     data = pickle.load(open(data_filename))

# #     f = pylab.figure(figsize= (12, 8))

    
# #     chains = [c for c in chains if type(c['scores']) != int]
# #     if "srm" or "ld" in plot_hypers_filename:
# #         irm.experiments.plot_chains_hypers(f, chains, data)

# #     f.savefig(plot_hypers_filename)

# # @transform(get_results, suffix(".samples"), [".hypers_individual.pdf"])
# # def plot_hypers(exp_results, (plot_hypers_filename,)):

# #     sample_d = pickle.load(open(exp_results))
# #     chains = sample_d['chains']
    
# #     exp = sample_d['exp']
# #     data_filename = exp['data_filename']
# #     data = pickle.load(open(data_filename))

# #     f = pylab.figure(figsize= (12, 8))

    
# #     chains = [c for c in chains if type(c['scores']) != int]
# #     if "srm" in plot_hypers_filename:
# #         # this is sort of a gross hack
# #         states = [c['state'] for c in chains]
# #         mu_hps = [s['relations']['R1']['hps']['mu_hp'] for s in states]
# #         ax = f.add_subplot(1, 3, 1)
# #         ax.hist(mu_hps)
# #         lambda_hps = [s['relations']['R1']['hps']['lambda_hp'] for s in states]
# #         ax = f.add_subplot(1, 3, 2)
# #         ax.hist(lambda_hps)

# #         alpha_hps = [s['domains']['d1']['hps']['alpha'] for s in states]
# #         ax = f.add_subplot(1, 3, 3)
# #         ax.hist(alpha_hps)


# #     f.savefig(plot_hypers_filename)


# # # @transform(get_results, suffix(".samples"), [".params.pdf"])
# # # def plot_params(exp_results, (plot_params_filename,)):
# # #     """ 
# # #     plot parmaeters
# # #     """
# # #     sample_d = pickle.load(open(exp_results))
# # #     chains = sample_d['chains']
    
# # #     exp = sample_d['exp']
# # #     data_filename = exp['data_filename']
# # #     data = pickle.load(open(data_filename))

# # #     chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]

# # #     best_chain_i = chains_sorted_order[0]
# # #     best_chain = chains[best_chain_i]
# # #     sample_latent = best_chain['state']
    
# # #     m = data['relations']['R1']['model']
# # #     ss = sample_latent['relations']['R1']['ss']
# # #     f = pylab.figure()
# # #     ax = f.add_subplot(3, 1, 1)
# # #     ax_xhist = f.add_subplot(3, 1, 2)
# # #     ax_yhist = f.add_subplot(3, 1, 3)
# # #     if m == "LogisticDistance":
# # #         mus_lambs = np.array([(x['mu'], x['lambda']) for x in ss.values()])
# # #         ax.scatter(mus_lambs[:, 0], mus_lambs[:, 1], edgecolor='none', 
# # #                    s=2, alpha=0.5)
# # #         ax.set_xlabel('mu')
# # #         ax.set_ylabel('labda')
# # #         ax.set_xlim(0, 150)
# # #         ax.set_ylim(0, 150)

# # #         ax_xhist.hist(mus_lambs[:, 0], bins=20)
# # #         ax_xhist.axvline(sample_latent['relations']['R1']['hps']['mu_hp'])
# # #         ax_yhist.hist(mus_lambs[:, 1], bins=40)
# # #         ax_yhist.axvline(sample_latent['relations']['R1']['hps']['lambda_hp'])

# # #     f.suptitle("chain %d for %s" % (0, plot_params_filename))
    
# # #     f.savefig(plot_params_filename)

# # CIRCOS_DIST_THRESHOLDS = [20, 40, 60]

# # @transform(get_results, suffix(".samples"), 
# #            [(".circos.%02d.svg" % d, 
# #              ".circos.%02d.small.svg" % d)  for d in range(len(CIRCOS_DIST_THRESHOLDS))])
# # def plot_circos_latent(exp_results, 
# #                        out_filenames):

# #     sample_d = pickle.load(open(exp_results))
# #     chains = sample_d['chains']
    
# #     exp = sample_d['exp']
# #     data_filename = exp['data_filename']
# #     data = pickle.load(open(data_filename))
# #     data_basename, _ = os.path.splitext(data_filename)
# #     meta = pickle.load(open(data_basename + ".meta"))

# #     meta_infile = meta['infile']
# #     print "meta_infile=", meta_infile

# #     d = pickle.load(open(meta_infile, 'r'))
# #     conn = d['conn_mat']
# #     cells = d['cells']
# #     cells['coarse'][cells['type_id'] == 58] = 'bc'

# #     cell_types = cells['type_id']

# #     chains = [c for c in chains if type(c['scores']) != int]
# #     CHAINN = len(chains)
# #     print "THERE ARE", CHAINN, "chains", [type(c['scores']) for c in chains]

# #     chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
# #     chain_pos = 0

# #     best_chain_i = chains_sorted_order[chain_pos]
# #     best_chain = chains[best_chain_i]
# #     sample_latent = best_chain['state']
# #     cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

# #     # soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
# #     # pos_vec = soma_positions['pos_vec'][cell_id_permutation]
# #     # print "Pos_vec=", pos_vec
# #     if 'R1' in data['relations']:
# #         model_name = data['relations']['R1']['model']
# #     else:
# #         model_name = None

# #     # this is potentially fun: get the ranges for each type
# #     TYPE_N = np.max(cell_types) + 1

# #     # df2 = pandas.DataFrame(index=np.arange(1, TYPE_N))
# #     # df2['des'] = type_metadata_df['coarse']
# #     # df2 = df2.fillna('other')
# #     # df2['id'] = df2.index.values.astype(int)
# #     # gc_mean_i = df2.groupby('des').mean().astype(int)
# #     # gc_min_i = df2.groupby('des').min().astype(int)
# #     # gc_max_i = df2.groupby('des').max().astype(int)



# #     TGT_CMAP = pylab.cm.gist_heat
# #     coarse_colors = {'other' : [210, 210, 210]}
# #     for n_i, n in enumerate(['gc', 'nac', 'mwac', 'bc']):
# #         coarse_colors[n] = colorbrewer.Set1[4][n_i]
# #     print "THE COARSE COLORS ARE", coarse_colors
    
# #     for fi, (circos_filename_main, circos_filename_small) in enumerate(out_filenames):
# #         CLASS_N = len(np.unique(cell_assignment))
        

# #         class_ids = sorted(np.unique(cell_assignment))

# #         custom_color_map = {}
# #         for c_i, c_v in enumerate(class_ids):
# #             c = np.array(pylab.cm.Set1(float(c_i) / CLASS_N)[:3])*255
# #             custom_color_map['ccolor%d' % c_v]  = c.astype(int)

# #         colors = np.linspace(0, 360, CLASS_N)
# #         color_str = ['ccolor%d' % int(d) for d in class_ids]

# #         for n, v in coarse_colors.iteritems():
# #             custom_color_map['true_coarse_%s' % n] = v

# #         circos_p = irm.plots.circos.CircosPlot(cell_assignment, 
# #                                                ideogram_radius="0.5r",
# #                                                ideogram_thickness="30p", 
# #                                                karyotype_colors = color_str, 
# #                                                custom_color_map = custom_color_map)

# #         if model_name == "LogisticDistance":
# #             v = irm.irmio.latent_distance_eval(CIRCOS_DIST_THRESHOLDS[fi], 
# #                                                sample_latent['relations']['R1']['ss'], 
# #                                                sample_latent['relations']['R1']['hps'], 
# #                                                model_name)
# #             thold = 0.60 
# #             ribbons = []
# #             links = []
# #             for (src, dest), p in v.iteritems():
# #                 if p > thold:
# #                     ribbons.append((src, dest, int(30*p)))
# #             circos_p.set_class_ribbons(ribbons)

# #         pos_min = 40
# #         pos_max = 120
# #         pos_r_min = 1.00
# #         pos_r_max = pos_r_min + 0.3
# #         ten_um_frac = 10.0/(pos_max - pos_min)

# #         circos_p.add_plot('scatter', {'r0' : '%fr' % pos_r_min, 
# #                                       'r1' : '%fr' % pos_r_max, 
# #                                       'min' : pos_min, 
# #                                       'max' : pos_max, 
# #                                       'glyph' : 'circle', 
# #                                       'glyph_size' : 8, 
# #                                       'color' : 'black',
# #                                       'stroke_thickness' : 0
# #                                       }, 
# #                           cells['x'], 
# #                           {'backgrounds' : [('background', {'color': 'vvlgrey', 
# #                                                             'y0' : pos_min, 
# #                                                             'y1' : pos_max})],  
# #                            'axes': [('axis', {'color' : 'vgrey', 
# #                                               'thickness' : 1, 
# #                                               'spacing' : '%fr' % ten_um_frac})]})

# #         # circos_p.add_plot('heatmap', {'r0' : '1.34r', 
# #         #                                 'r1' : '1.37r', 
# #         #                                 'min' : 0, 
# #         #                                 'max' : 72, 
# #         #                               'stroke_thickness' : 0, 
# #         #                               'color' : ",".join(true_color_list) }, 
# #         #                   cell_types)
        
# #         # types_sparse = np.array(cells['type_id'], dtype=np.float32)
# #         # types_sparse[types_sparse <72] = np.nan
 
# #         # circos_p.add_plot('scatter', {'r0' : '1.8r', 
# #         #                               'r1' : '1.9r', 
# #         #                               'min' : 70, 
# #         #                               'max' : 78, 
# #         #                               'gliph' : 'circle', 
# #         #                               'color' : 'black', 
# #         #                               'stroke_thickness' : 0}, 
# #         #                   types_sparse, 
# #         #                   {'backgrounds' : [('background', {'color': 'vvlblue', 
# #         #                                                     'y0' : 70, 
# #         #                                                     'y1' : 78})],  
# #         #                    'axes': [('axis', {'color' : 'vgrey', 
# #         #                                       'thickness' : 1, 
# #         #                                       'spacing' : '%fr' % (0.1)})]})

        
# #         # circos_p.add_plot('heatmap', {'r0' : '1.7r', 
# #         #                              'r1' : '1.8r'}, 
# #         #                  cells['coarse'])

# #         # always plot the cell depth
# #         # compute cell depth histograms per-row

# #         X_HIST_BINS = np.linspace(60, 120, 20)
# #         hists = np.zeros((len(cells), len(X_HIST_BINS)-1))
# #         for cell_i, (cell_id, cell) in enumerate(cells.iterrows()):
# #             h, e = np.histogram(cell['contact_x_list'], X_HIST_BINS)
# #             hists[cell_i] = h

# #         for bi, b in enumerate(X_HIST_BINS[:-1]):
# #             width = 0.4/20.
# #             start = 1.3 + width*bi
# #             end = start + width
# #             circos_p.add_plot('heatmap', {'r0' : '%fr' % start, 
# #                                           'r1' : '%fr' % end, 
# #                                           'stroke_thickness' : 0, 
# #                                           'color' : 'greys-6-seq'}, 
# #                               hists[:, bi])


# #         # f_color_legend = pylab.figure()
# #         # ax_color_legend = f_color_legend.add_subplot(1, 1, 1)

# #         # x = np.zeros((TYPE_N, 20))
# #         # for i in range(10):
# #         #     x[:, i] = np.arange(TYPE_N)
# #         # for n in ['gc', 'nac', 'mwac', 'bc', 'other']:
# #         #     print gc_min_i
# #         #     x[gc_min_i.ix[n]:gc_max_i.ix[n]+1, 10:] = gc_mean_i.ix[n]
# #         #     ax_color_legend.plot([10, 20], [gc_max_i.ix[n], gc_max_i.ix[n]])
# #         # ax_color_legend.imshow(x, cmap=TGT_CMAP, interpolation='nearest')
# #         # ax_color_legend.axvline(10, c='k')
# #         # ax_color_legend.set_xticks([])
# #         # f_color_legend.savefig(color_legend_filename)

# #         #COARSE COLOR TYPES
# #         plot_truth_coarse = True

# #         if plot_truth_coarse:
# #             print "TYPE_N=", TYPE_N
# #             type_color_map = {'gc' : 0, 
# #                               'nac' : 1, 
# #                               'mwac' : 2, 
# #                               'bc' : 3, 
# #                               'other' : 4, 
# #                               None : 4}

# #             # pick colors
# #             colors = ['true_coarse_%s' % s for s in ['gc', 'nac', 'mwac', 'bc', 'other']]

# #             circos_p.add_plot('heatmap', {'r0' : '1.7r', 
# #                                           'r1' : '1.8r', 
# #                                           'min' : 0, 
# #                                           'max' : 4, 
# #                                           'stroke_thickness' :0,
# #                                           'color': ",".join(colors)}, 
# #                               [type_color_map[c] for c in cells['coarse']])
# #             # circos_p.add_plot('text', {'r0' : '1.9r', 
# #             #                            'r1' : '2.0r', 
# #             #                            'label_size' : '7p'}, 
# #             #                   cells['type_id'])
# #         else:
# #             ### FINE TRUE TYPES
# #             circos_p.add_plot('heatmap', {'r0' : '1.7r', 
# #                                           'r1' : '1.8r', 
# #                                           'min' : 0, 
# #                                           'max' : 80, 
# #                                           'stroke_thickness' :0,
# #                                           }, 
# #                               cells['type_id'])


# #         # circos_p.add_plot('scatter', {'r0' : '1.01r', 
# #         #                               'r1' : '1.10r', 
# #         #                               'min' : 0, 
# #         #                               'max' : 3, 
# #         #                               'gliph' : 'square', 
# #         #                               'color' : 'black', 
# #         #                               'stroke_thickness' : 0}, 
# #         #                   [type_lut[i] for i in cell_types])



# #         irm.plots.circos.write(circos_p, circos_filename_main)
        
# #         circos_p = irm.plots.circos.CircosPlot(cell_assignment, ideogram_radius="0.5r", 
# #                                                ideogram_thickness="80p", 
# #                                                karyotype_colors = color_str, 
# #                                                custom_color_map = custom_color_map)
        
# #         if model_name == "LogisticDistance":
# #             v = irm.irmio.latent_distance_eval(CIRCOS_DIST_THRESHOLDS[fi], 
# #                                                sample_latent['relations']['R1']['ss'], 
# #                                                sample_latent['relations']['R1']['hps'], 
# #                                                model_name)
# #             thold = 0.50 
# #             ribbons = []
# #             links = []
# #             for (src, dest), p in v.iteritems():
# #                 if p > thold:
# #                     ribbons.append((src, dest, int(30*p)))
# #             circos_p.set_class_ribbons(ribbons)
                                            
# #         irm.plots.circos.write(circos_p, circos_filename_small)
        
# # @transform(get_results, suffix(".samples"), 
# #            ".somapos.pdf")

# # def plot_clustered_somapos(exp_results, 
# #                            out_filename):

# #     sample_d = pickle.load(open(exp_results))
# #     chains = sample_d['chains']
    
# #     exp = sample_d['exp']
# #     data_filename = exp['data_filename']
# #     data = pickle.load(open(data_filename))
# #     data_basename, _ = os.path.splitext(data_filename)
# #     meta = pickle.load(open(data_basename + ".meta"))

# #     meta_infile = meta['infile']
# #     print "meta_infile=", meta_infile

# #     d = pickle.load(open(meta_infile, 'r'))
# #     conn = d['conn_mat']
# #     cells = d['cells']

# #     cell_types = cells['type_id']

# #     chains = [c for c in chains if type(c['scores']) != int]
# #     CHAINN = len(chains)
# #     print "THERE ARE", CHAINN, "chains", [type(c['scores']) for c in chains]

# #     chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
# #     chain_pos = 0

# #     best_chain_i = chains_sorted_order[chain_pos]
# #     best_chain = chains[best_chain_i]
# #     sample_latent = best_chain['state']
# #     cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

    
# #     cells['cluster'] = cell_assignment
# #     print cells.head()
# #     print cells

# #     class_ids = sorted(np.unique(cell_assignment))
# #     CLASS_N = len(class_ids)

# #     custom_color_map = {}
# #     for c_i, c_v in enumerate(class_ids):
# #         c = np.array(pylab.cm.Set1(float(c_i) / CLASS_N)[:3])*255
# #         custom_color_map[c_v]  = c.astype(int)

# #     f = pylab.figure(figsize=(12, 8))
# #     ax = f.add_subplot(1, 1, 1)



# #     ca = irm.util.canonicalize_assignment(cell_assignment)
# #     # build up the color rgb
# #     cell_colors = np.zeros((len(ca), 3))
# #     for ci, c in enumerate(ca):
# #         cell_colors[ci] = pylab.cm.Set1(float(c) / CLASS_N)[:3]
# #     ax.scatter(cells['y'], cells['z'], edgecolor='none', 
# #                c = cell_colors, s=60)
# #     ax.set_ylim(0, 85)
# #     ax.set_xlim(5, 115)
# #     ax.set_aspect(1.0)
# #     ax.plot([10, 20], [3, 3], linewidth=5, c='k')
# #     ax.set_xticks([])
# #     ax.set_yticks([])

# #     f.savefig(out_filename)

# # # @transform(get_results, suffix(".samples"), 
# # #            ".truth_latent.pdf" )
# # # def plot_truth_latent(exp_results, 
# # #                       out_filename):

# # #     sample_d = pickle.load(open(exp_results))
# # #     chains = sample_d['chains']
    
# # #     exp = sample_d['exp']
# # #     data_filename = exp['data_filename']
# # #     data = pickle.load(open(data_filename))
# # #     data_basename, _ = os.path.splitext(data_filename)
# # #     meta = pickle.load(open(data_basename + ".meta"))

# # #     meta_infile = meta['infile']
# # #     print "meta_infile=", meta_infile

# # #     d = pickle.load(open(meta_infile, 'r'))
# # #     conn = d['dist_matrix']['link']
# # #     cell_id_permutation = d['cell_id_permutation']
    
# # #     dist_matrix = d['dist_matrix']
# # #     orig_data = pickle.load(open(d['infile']))
# # #     cell_types = d['types'][:len(conn)]
    
# # #     type_metadata_df = pickle.load(open("type_metadata.pickle", 'r'))['type_metadata']

# # #     chains = [c for c in chains if type(c['scores']) != int]
# # #     CHAINN = len(chains)

# # #     chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
# # #     chain_pos = 0

# # #     best_chain_i = chains_sorted_order[chain_pos]
# # #     best_chain = chains[best_chain_i]
# # #     sample_latent = best_chain['state']
# # #     cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

# # #     soma_positions = pickle.load(open('soma.positions.pickle', 'r'))
# # #     pos_vec = soma_positions['pos_vec'][cell_id_permutation]
# # #     print "Pos_vec=", pos_vec
# # #     model_name = data['relations']['R1']['model']

# # #     # this is potentially fun: get the ranges for each type
# # #     TYPE_N = np.max(cell_types) + 1

# # #     df2 = pandas.DataFrame(index=np.arange(1, TYPE_N))
# # #     df2['des'] = type_metadata_df['coarse']
# # #     df2 = df2.fillna('other')
# # #     df2['id'] = df2.index.values.astype(int)
# # #     gc_mean_i = df2.groupby('des').mean().astype(int)
# # #     gc_min_i = df2.groupby('des').min().astype(int)
# # #     gc_max_i = df2.groupby('des').max().astype(int)

# # #     TGT_CMAP = pylab.cm.gist_heat
# # #     coarse_colors = {'other' : [210, 210, 210]}
# # #     for n_i, n in enumerate(['gc', 'nac', 'mwac', 'bc']):
# # #         coarse_colors[n] = colorbrewer.Set1[4][n_i]
# # #     print "THE COARSE COLORS ARE", coarse_colors

# # #     type_color_map = {'gc' : 0, 
# # #                       'nac' : 1, 
# # #                       'mwac' : 2, 
# # #                       'bc' : 3, 
# # #                       'other' : 4}

# # #     df = pandas.DataFrame({'cell_id' : cell_id_permutation, 
# # #                            'cell_type' : cell_types, 
# # #                            'cluster' : cell_assignment})
# # #     df = df.join(df2, on='cell_type')
# # #     print df.head()

# # #     CLASS_N = len(np.unique(cell_assignment))
# # #     f = pylab.figure(figsize=(8, 11))
# # #     fid = open(out_filename + '.html', 'w')

# # #     # compute the axes positions
# # #     COL_NUMBER = 4
# # #     COL_SPACE = 1.0 / COL_NUMBER
# # #     COL_WIDTH = COL_SPACE - 0.03
# # #     COL_PRE = 0.02
# # #     ROW_CONTRACT = 0.05
# # #     ROW_PRE = 0.02
# # #     ROW_SPACE = 0.005
# # #     ROW_HEIGHT_MIN = 0.015
# # #     s = df['cluster'].value_counts()

# # #     a = irm.util.multi_napsack(COL_NUMBER, np.array(s))
# # #     CLUST_MAX_PER_COL = len(a[0])
# # #     VERT_SCALE = 0.95 -  (CLUST_MAX_PER_COL+4) *ROW_SPACE

# # #     MAX_LEN = np.sum([np.array(s)[ai] for ai in a[0]])

# # #     for col_i, col in enumerate(a):
# # #         pos = 0
# # #         for row_pos in col:
# # #             cluster_id = s.index.values[row_pos]
# # #             sub_df = df[df['cluster'] == cluster_id]
# # #             sub_df = sub_df.sort('cell_type')
# # #             height = len(sub_df) / float(MAX_LEN) * VERT_SCALE
# # #             height = np.max([height, ROW_HEIGHT_MIN])
# # #             ax = f.add_axes([COL_PRE + col_i * COL_SPACE, 
# # #                              1.0 - pos - height - ROW_PRE, 
# # #                              COL_WIDTH, height])

# # #             CN = len(sub_df)
            
# # #             for i in range(CN):
# # #                 ax.axhline(i, c='k', alpha=0.05)

# # #             colors = [np.array(coarse_colors[ct])/255.0 for ct in sub_df['des']]
# # #             ax.scatter(sub_df['cell_type'], np.arange(CN), 
# # #                        c= colors, s=15,
# # #                        edgecolor='none')
# # #             # optionally plot text
# # #             for i in range(CN):
# # #                 t = sub_df.irow(i)['cell_type']
# # #                 xpos = 1
# # #                 hl = 'left'
# # #                 if t < 30:
# # #                     xpos = TYPE_N-2
# # #                     hl = 'right'
# # #                 ax.text(xpos, i, "%d" % sub_df.index.values[i], 
# # #                         verticalalignment='center', fontsize=3.5,
# # #                         horizontalalignment=hl)
            
# # #             ax.set_yticks([])
# # #             ax.grid(1)
# # #             ax.set_xlim(0, TYPE_N)
# # #             if CN > 3:
# # #                 ax.set_ylim(-1, CN+0.5)
# # #             else:
# # #                 ax.set_ylim(-1, 3)

# # #             ax.set_xticks([10, 20, 30, 40, 50, 60, 70])


# # #             for tick in ax.xaxis.iter_ticks():
# # #                 if pos == 0 :
# # #                     tick[0].label2On = True                    
# # #                 tick[0].label1On = False
# # #                 tick[0].label2.set_rotation('vertical')
# # #                 tick[0].label2.set_fontsize(6) 
# # #             pos += height + ROW_SPACE

# # #     # fid.write(group.to_html())
# # #     fid.close()

# # #     f.savefig(out_filename)

# # def compute_cluster_metrics_raw(chains, cells):

# #     all_chains = []
# #     for chain_i, chain in enumerate(chains):

# #         sample_latent = chain['state']
# #         cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])
# #         ca = irm.util.canonicalize_assignment(cell_assignment)

# #         cells['cluster'] = ca

# #         canon_true_fine = irm.util.canonicalize_assignment(cells['type_id'])
# #         canon_true_coarse = irm.util.canonicalize_assignment(cells['coarse'])



# #         ari = metrics.adjusted_rand_score(canon_true_fine, ca)
# #         ari_coarse = metrics.adjusted_rand_score(canon_true_coarse, ca)

# #         ami = metrics.adjusted_mutual_info_score(canon_true_fine, ca)
# #         ami_coarse = metrics.adjusted_mutual_info_score(canon_true_coarse, ca)


# #         jaccard = rand.compute_jaccard(canon_true_fine, ca)
# #         jaccard_coarse = rand.compute_jaccard(canon_true_coarse, ca)

# #         ss = rand.compute_similarity_stats(canon_true_fine, ca)

# #         # other statistics 

# #         # cluster count

# #         # average variance x
# #         vars = cells.groupby('cluster').var()
# #         # average variance y
# #         # average variance z

# #         chain_info = {'ari' : ari, 
# #                      'ari_coarse' : ari_coarse, 
# #                      'ami' : ami, 
# #                      'ami_coarse' : ami_coarse, 
# #                      'jaccard' : jaccard, 
# #                      'jaccard_coarse' : jaccard_coarse,
# #                      'n11' : ss['n11'], 
# #                      'vars' : vars, 
# #                       'cluster_n' : len(np.unique(cells['cluster'])),
# #                       'chain_i' : chain_i, 
# #                       'score' : chain['scores'][-1],
# #                       'df' : cells, 
# #                      }
# #         all_chains.append(chain_info)
# #     df = pandas.DataFrame(all_chains)
# #     return df

# # @transform(get_results,
# #            suffix(".samples"), 
# #            ".cluster_metrics.pickle" )
# # def compute_cluster_metrics(exp_results, 
# #                       out_filename):

# #     sample_d = pickle.load(open(exp_results))
# #     chains = sample_d['chains']
    
# #     exp = sample_d['exp']
# #     data_filename = exp['data_filename']
# #     data = pickle.load(open(data_filename))
# #     data_basename, _ = os.path.splitext(data_filename)
# #     meta = pickle.load(open(data_basename + ".meta"))

# #     meta_infile = meta['infile']
# #     print "meta_infile=", meta_infile

# #     d = pickle.load(open(meta_infile, 'r'))
# #     conn = d['conn_mat']
# #     cells = d['cells']

    
# #     chains = [c for c in chains if type(c['scores']) != int]
# #     df = compute_cluster_metrics_raw(chains, cells)
# #     df['filename'] = exp_results
# #     pickle.dump(df, open(out_filename, 'w'))


# # # @files(["data/retina.1.simple_bb.data-fixed_4-anneal_slow_400.samples", 
# # #         "data/retina.1.data.pickle"],
# # #        "data/retina.1.simple_bb.data-fixed_4-anneal_slow_400.cluster_metrics.pickle" )
# # # def compute_cluster_metrics_bb((samples, data), 
# # #                                out_filename):

# # #     sample_d = pickle.load(open(samples))
# # #     chains = sample_d['chains']

# # #     d = pickle.load(open(data, 'r'))
# # #     cells = d['cells']
    
# # #     chains = [c for c in chains if type(c['scores']) != int]
# # #     df = compute_cluster_metrics_raw(chains, cells)
# # #     df['filename'] = samples

# # #     pickle.dump(df, open(out_filename, 'w'))


# # @merge([compute_cluster_metrics, 
# #         #compute_cluster_metrics_bb
# #     ], 
# #        ("cluster_metrics.pickle", 'cluster.metrics.html'))
# # def merge_cluster_metrics(infiles, (outfile_pickle, outfile_html)):
# #     res = []
# #     v_df = []
# #     for infile in infiles:
# #         df = pickle.load(open(infile, 'r'))
        
# #         res.append(df)

# #     # # add in the two others
# #     # fine_vars = df.copy().groupby('type_id').var()
# #     # fine_vars['filename'] = "truth.fine"

# #     # coarse_vars = df.copy().groupby('coarse').var()
# #     # coarse_vars['filename'] = "truth.coarse"
# #     # print coarse_vars
# #     # v_df.append(fine_vars)
# #     # v_df.append(coarse_vars)

# #     clust_df = pandas.concat(res)
# #     print clust_df
# #     #var_df = pandas.concat(v_df)
# #     del clust_df['df']
# #     clust_df = clust_df.set_index([np.arange(len(clust_df))])

# #     pickle.dump({'clust_df' : clust_df, 
# #                  #'var_df' : var_df
# #              },
# #                 open(outfile_pickle, 'w'))

# #     fid = open(outfile_html, 'w')
# #     #fid.write(clust_df.to_html())
# #     #fid.write(var_df.to_html())
# #     fid.close()

# # # @files(merge_cluster_metrics, ("spatial_var.pdf", "spatial_var.txt"))
# # # def plot_cluster_vars((infile_pickle, infile_rpt), (outfile_plot, outfile_rpt)):
# # #     d = pickle.load(open(infile_pickle, 'r'))

# # #     var_df = d['var_df']
# # #     var_df = var_df[np.isfinite(var_df['x'])]
# # #     var_df = var_df[np.isfinite(var_df['y'])]
# # #     var_df = var_df[np.isfinite(var_df['z'])]
# # #     tgts = [('Infintite Stochastic Block Model',
# # #              "1.2.bb.0.0.data-fixed_20_100-anneal_slow_400", 'r', None), 
# # #             ('Infinite Spatial-Relational Model', 
# # #              "1.2.ld.0.0.data-fixed_20_100-anneal_slow_400", 'b', None), 
# # #             ('Finite SBM, K=12', 
# # #              "1.2.bb.0.0.data-fixed_20_12-anneal_slow_fixed_400", 'g', None), 
# # #             ('Truth (fine)', 'truth.fine' ,'k', {'linewidth' : 2, 
# # #                                                  'linestyle' : '--'}), 
# # #             ('Truth (coarse)', 'truth.coarse', 'k', {'linewidth' : 4}),
# # #         ]

# # #     f = pylab.figure(figsize=(8,6))
# # #     ax = f.add_subplot(1, 1, 1)
# # #     normed = True
# # #     report_fid = open(outfile_rpt, 'w')
# # #     for t_i, (tgt_name, tgt_fname, c, args) in enumerate(tgts):
# # #         var_df_sub = var_df[var_df['filename'].str.contains(tgt_fname)]

# # #         s = np.sqrt(var_df_sub['y'] + var_df_sub['z'])
# # #         mean = np.mean(s)
# # #         std = np.std(s)
        
# # #         bins = np.linspace(0, 60, 40)

# # #         if 'Truth' not in tgt_name:
# # #             ax.hist(s, bins=bins, 
# # #                     normed=normed, color=c, label=tgt_name)
# # #         else:
# # #             hist, edge = np.histogram(s, bins=bins, normed=normed)
# # #             centers = bins[:-1] + (bins[1] - bins[0])/2.
            
# # #             ax.plot(centers, hist, c=c, label=tgt_name, 
# # #                     **args)
# # #         report_fid.write("%s: mean = %f std=%f \n" % (tgt_name, mean, std))

# # #     ax.set_xlim(0, 60)
# # #     ax.set_xlabel("std. dev. of type (um)")
# # #     ax.set_ylabel("fraction")
# # #     ax.legend(loc="upper left")
# # #     ax.set_title("spatial distribution of type")
# # #     # f = pylab.figure(figsize=(6, 8))
# # #     # for i, v in enumerate(['x', 'y', 'z']):
# # #     #     ax = f.add_subplot(3, 1, i + 1)
# # #     #     vars = []
# # #     #     for t_i, (tgt_name, tgt_fname) in enumerate(tgts):
# # #     #         var_df_sub = var_df[var_df['filename'].str.contains(tgt_fname)]
# # #     #         vars.append(np.sqrt(var_df_sub[v]))

# # #     #     ax.boxplot(vars)
# # #     #     ax.set_xticklabels([x[0] for x in tgts])
# # #     #     ax.set_ylabel("standard dev")
# # #     #     ax.set_title(v)


# # #     f.tight_layout()
# # #     f.savefig(outfile_plot)

# # # @files(merge_cluster_metrics, ("ari_vs_cluster.pdf", "ari_vs_cluster.html"))
# # # def plot_cluster_aris((infile_pickle, infile_report), 
# # #                       (outfile_plot, outfile_rpt)):
# # #     d = pickle.load(open(infile_pickle, 'r'))

# # #     tgt_config = "1.1"

# # #     clust_df = d['clust_df']
# # #     clust_df = clust_df[clust_df['filename'].str.contains("retina.%s" % tgt_config)]
# # #     clust_df['finite'] = clust_df['filename'].str.contains("_slow_fixed_400")


# # #     finite_df = clust_df[clust_df['finite']]

# # #     ld_df = clust_df[clust_df['filename'].str.contains("ld")]
# # #     bb_df = clust_df[clust_df['filename'].str.contains("_slow_400")]
# # #     bb_df = bb_df[bb_df['filename'].str.contains("bb")]

# # #     f = pylab.figure()
# # #     ax = f.add_subplot(1, 1, 1)

# # #     index = 'ari_coarse'
# # #     ax.scatter(finite_df['cluster_n'], finite_df[index], s=50, c='b', label="SBM")


# # #     ax.scatter(bb_df['cluster_n'], bb_df[index], s=90, c='g', label="iSBM")
# # #     ax.scatter(ld_df['cluster_n'], ld_df[index], s=90, c='r', label="iSRM")

# # #     ax.set_xlabel("number of found types")
# # #     ax.set_ylabel("adjusted rand index")
# # #     ax.set_xlim(0, 120)
# # #     ax.grid()
# # #     ax.legend()

# # #     f.savefig(outfile_plot)

# # #     fid = open(outfile_rpt, 'w')
# # #     fid.write(clust_df.to_html())
# # #     fid.close()
              
if __name__ == "__main__":    
    pipeline_run([data_preprocess,
                  #data_celegans_adj,
                  create_latents_2r_paramed,
                  #create_latents_srm, 
                  #create_latents_srm_clist_xsoma,
                  spark_run_experiments,
                  get_cvdata,
                  get_samples,
                  samples_organize,
                  cvdata_organize,
                  cv_collate_predlinks_assign
                  #cv_data_organize,
                  #plot_hypers,
                  #plot_circos_latent, 
                  #compute_cluster_metrics, 
                  #compute_cluster_metrics_bb, 
                  #merge_cluster_metrics,
                  #plot_data_raw, 
                  # data_retina_adj_count, 
              # create_inits, 
              # plot_scores_z, 
                  #plot_best_cluster_latent, 
              # #plot_hypers, 
              # plot_latents_ld_truth, 
              # plot_params, 
              # create_latents_ld_truth, 
              # plot_circos_latent, 
                  #plot_z_matrix, 
                  #plot_clustered_somapos,
              #plot_cluster_vars, 
              #plot_cluster_aris, 
              ], multiprocess=1) # , multiprocess=3)
    
