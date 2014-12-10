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
import multyvac

import irm
import irm.data
import util
from irm import rand
import predutil
import boto
from pyspark import SparkContext
import cvpipelineutil as cv
import sparkutil

# def saveAsPickleFile(obj, tgtdir, overwrite=True):
#     try:
#         if overwrite:
#             shutil.rmtree(tgtdir)
#     except OSError:
#         pass
    
#     obj.saveAsPickleFile(tgtdir)



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
    ('srm.00', 'cv_nfold_2', 'debug_2_100', 'debug_2'), 

    #('srm', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
    #('sbmnodist', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
    #('lpcm', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
    #('mm', 'cv_nfold_10', 'fixed_20_100', 'anneal_slow_1000'), 
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
    'debug_2' : {'ITERS' : 2, 
                  'kernels': irm.runner.default_kernel_anneal(1.0, 2)
              },
    'debug_200' : {'ITERS' : 200, 
                  'kernels': irm.runner.default_kernel_anneal(1.0, 160)
              }
    }


# 


def srm_params():
    for seed in range(10):
        filename = "srm_%02d.sourcedata" % seed 
    yield None, td(filename), seed

@files(srm_params)
def create_data_srm(infile, outfile, seed):
    np.random.seed(seed)
    conn_config = {}
    CLASS_N = 4
    SIDE_N = 10
    nonzero_frac = 0.8
    JITTER = 0.01


    for c1 in range(CLASS_N):
        for c2 in range(CLASS_N):
            if np.random.rand() < nonzero_frac:
                conn_config[(c1, c2)] = (np.random.uniform(1.0, 4.0), 
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


def sbmnodist_params():
    for seed in range(10):
        filename = "sbmnodist_%02d.sourcedata" % seed 
    yield None, td(filename), seed

@files(sbmnodist_params)
def create_data_sbm_nodist(infile, outfile, seed):
    np.random.seed(seed)
    conn_config = {}
    CLASS_N = 4
    SIDE_N = 10
    nonzero_frac = 0.8
    JITTER = 0.01


    for c1 in range(CLASS_N):
        for c2 in range(CLASS_N):
            if np.random.rand() < nonzero_frac:
                conn_config[(c1, c2)] = (100, # guarantee that we always have conn
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



def lpcm_params():
    for seed in range(10):
        filename = "lpcm_%02d.sourcedata" % seed 
    yield None, td(filename), seed

@files(lpcm_params)
def create_data_latent_position(infile, outfile, seed):
    np.random.seed(seed)

    LATENT_D = 3
    CLASS_N = 4
    SIDE_N = 10 # cells per class
    latent_class_variance = 5.0
    # draw class_n isotropic gaussians

    class_params = {}
    for i in range(CLASS_N):
        p = {'mu' : np.random.normal(0, 1, LATENT_D), 
             'var' : np.eye(LATENT_D) * latent_class_variance}
        class_params[i] = p 
    

    nonzero_frac = 0.8
    JITTER = 0.01

    node_base = irm.data.generate.create_nodes_grid(SIDE_N, CLASS_N, JITTER)
    NODE_N = len(node_base)
    nodes_with_meta = np.zeros(NODE_N, 
                     dtype=[('class',  np.uint32), 
                            ('pos' ,  np.float32, (3, )),
                            ('latentpos', np.float32, (LATENT_D, ))])
    nodes_with_meta['class'] = node_base['class']
    nodes_with_meta['pos'] = node_base['pos']

    for i in range(NODE_N):
        c = nodes_with_meta[i]['class']
        latentpos = np.random.multivariate_normal(class_params[c]['mu'],
                                                   class_params[c]['var'])
        nodes_with_meta[i]['latentpos'] = latentpos
        

    # now the connectivity
    connectivity = np.zeros((NODE_N, NODE_N), dtype=np.uint8)
                             
    for i in range(NODE_N):
        for j in range(NODE_N):
            lp1 = nodes_with_meta['latentpos'][i]
            lp2 = nodes_with_meta['latentpos'][j]
            p1 = nodes_with_meta['pos'][i]
            p2 = nodes_with_meta['pos'][j]
            mu = dist(lp1, lp2)
            d = dist(p1, p2)
            # now the connectivity function is that the mu is set by
            p = irm.util.logistic(d, mu, 1./mu)
            
            connectivity[i, j] = np.random.rand() < p

            
    ai = np.random.permutation(len(nodes_with_meta))
    nodes_with_meta = nodes_with_meta[ai]
    connectivity = connectivity[ai]
    connectivity = connectivity[:, ai]
    
    pickle.dump({'nodes_with_meta' : nodes_with_meta,
                 'connectivity' : connectivity,
                 'jitter' : JITTER},
                open(outfile, 'w'))



def mm_params():
    for seed in range(10):
        filename = "mm_%02d.sourcedata" % seed 
    yield None, td(filename), seed

@files(mm_params)
def create_data_mixed_membership(infile, outfile, seed):
    """
    Each cell can have some subset of 4 properties, and there exist
    arbitrary predicates that dictate whether or not it connects
    to other cells. 

    This is really equivalent to just 2^4 possible latent states, 
    but with a simpler structure on connectivity
    """
    
    np.random.seed(seed)
    conn_config = {}
    CLASS_ATTR_N = 4
    CLASS_N = 2**CLASS_ATTR_N
    SIDE_N = 5
    nonzero_frac = 0.4
    JITTER = 0.01


    for c1 in range(CLASS_N):
        for c2 in range(CLASS_N):
            if np.random.rand() < nonzero_frac:
                conn_config[(c1, c2)] = (np.random.uniform(1.0, 4.0), 
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


def data_generator():
    for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
        yield (td(data_name + ".sourcedata"),
               [td(data_name + ".data"),
                td(data_name + ".latent"),
                td(data_name + ".meta")])

        
@follows(create_data_srm)
@follows(create_data_sbm_nodist)
@follows(create_data_latent_position)
@follows(create_data_mixed_membership)
@files(data_generator)
def create_data_latent(infile, (data_filename, latent_filename, 
                                meta_filename)):


    """
    Input is a list of nodes, cell positions, and the connectivity
    """

    d = pickle.load(open(infile, 'r'))
    nodes_with_meta =  d['nodes_with_meta']
    connectivity = d['connectivity']


    conn_and_dist = np.zeros(connectivity.shape, 
                             dtype=[('link', np.uint8), 
                                    ('distance', np.float32)])
    print "conn_and_dist.dtype", conn_and_dist.dtype
    for ni, foo in enumerate(nodes_with_meta):
        for nj, bar in enumerate(nodes_with_meta):
            conn_and_dist[ni, nj]['link'] = connectivity[ni, nj]
            conn_and_dist[ni, nj]['distance'] = dist(nodes_with_meta[ni]['pos'],
                                                     nodes_with_meta[nj]['pos'])

    
    model_name = "LogisticDistance"
    latent, data = irm.irmio.default_graph_init(conn_and_dist, model_name)
    latent['domains']['d1']['assignment'] = nodes_with_meta['class']
    
    mulamb = 10.0
    p_max = 0.95

    HPS = {'mu_hp' : mulamb,
           'lambda_hp' : mulamb,
           'p_min' : 0.01, 
           'p_max' : p_max}

    latent['relations']['R1']['hps'] = HPS




    pickle.dump(latent, open(latent_filename, 'w'))

    pickle.dump(data, open(data_filename, 'w'))
    
    pickle.dump({'infile' : infile,
                 'conn_and_dist' : conn_and_dist, 
                 }, open(meta_filename, 'w'))


    

        
@follows(create_data_latent)
@files(list(cv.experiment_generator(EXPERIMENTS, CV_CONFIGS,
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


@subdivide(get_cvdata, formatter(".+/(?P<base>.*).cvdata.pickle"), 
           "{path[0]}/{base[0]}.samples.organized/*/cv.data",
           # Output parameter: Glob matches any number of output file names
            "{path[0]}/{base[0]}.samples.organized")          # Extra parameter:  Append to this for output file names
def cvdata_organize(input_file, output_files, output_file_name_root):
    print "input_file=", input_file
    a = pickle.load(open(input_file, 'r'))
    for di, d in enumerate(a):
        key = d[0]
        a = key.split('.')[1]
        data, meta = d[1]
        pickle.dump(data, open(os.path.join(output_file_name_root, a, "cv.data"), 'w'))
        pickle.dump(meta, open(os.path.join(output_file_name_root, a, "cv.meta"), 'w'))


@follows(samples_organize)
@follows(cvdata_organize)
@collate("sparkdata/*.samples.organized/*",
         regex(r"(sparkdata/.+.samples.organized)/(\d+)"),
         #[td(r"\1-\2\4.predlinks"), td(r"\1-\2\4.assign")])
         [r"\1/predlinks.pickle", r'\1/assign.pickle'])
def cv_collate_predlinks_assign(cv_dirs, (predlinks_outfile,
                                          assign_outfile)):
    """
    aggregate across samples and cross-validations
    """
    print "THIS IS A RUN", predlinks_outfile
    for d in cv_dirs:
        print "cv dir", d

    
    base = os.path.dirname(cv_dirs[0])[:-len('.samples.organized')]
    s = base.split('-')

    input_basename = s[0]
    input_data = pickle.load(open(input_basename + ".data"))
    input_latent = pickle.load(open(input_basename + ".latent"))
    data_conn = input_data['relations']['R1']['data']
    model_name= input_data['relations']['R1']['model']


    N = len(data_conn)

    true_assign = input_latent['domains']['d1']['assignment']
    print "for", input_basename, "there are", len(np.unique(true_assign)), "classes"
    if model_name == "BetaBernoulliNonConj":
        truth_mat = data_conn
    elif model_name == "LogisticDistance":
        truth_mat = data_conn['link']
        # truth_mat_t_idx = np.argwhere(truth_mat.flatten() > 0).flatten()

    predlinks_outputs = []
    assignments_outputs = []


    N = len(data_conn)
    
    for cv_dir in cv_dirs:
        cv_data = pickle.load(open(os.path.join(cv_dir, 'cv.data'), 'r'))
        # get the cv idx for later use
        cv_idx = int(os.path.basename(cv_dir))
        

    
        
        # FIGURE OUT WHICH ENTRIES WERE MISSING
        heldout_idx = np.argwhere((cv_data['relations']['R1']['observed'].flatten() == 0)).flatten()
        heldout_true_vals = truth_mat.flatten()[heldout_idx]
        
        heldout_true_vals_t_idx = np.argwhere(heldout_true_vals > 0).flatten()
        heldout_true_vals_f_idx = np.argwhere(heldout_true_vals == 0).flatten()
        
        sample_file_str =os.path.join(cv_dir, r"[0-9]*.pickle")
        print "files are"
        for sample_name in glob(sample_file_str):
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
                                          'f_tot' : len(heldout_true_vals_f_idx)
                                      })
            assignments_outputs.append({'chain_i' : chain_i,
                                        'score' : scores[-1],
                                        'cv_idx' : cv_idx, 
                                        'true_assign' : true_assign,
                                        'assign' : irm_latent_samp['domains']['d1']['assignment']})
                                            


    predlinks_df = pandas.DataFrame(predlinks_outputs)
    pickle.dump({'df' : predlinks_df}, 
                 open(predlinks_outfile, 'w'))

    a_df = pandas.DataFrame(assignments_outputs)
    pickle.dump({'df' : a_df}, 
                 open(assign_outfile, 'w'))

                  

@transform(cv_collate_predlinks_assign, suffix("predlinks.pickle"), "roc.pdf")
def plot_predlinks_roc(infile, outfile):
    preddf = pickle.load(open(infile[0], 'r'))['df']
    preddf['tp'] = preddf['t_t'] / preddf['t_tot']
    preddf['fp'] = preddf['f_t'] / preddf['f_tot']
    preddf['frac_wrong'] = 1.0 - (preddf['t_t'] + preddf['f_f']) / (preddf['t_tot'] + preddf['f_tot'])

    f = pylab.figure(figsize=(4, 4))
    ax = f.add_subplot(1, 1, 1)
    
    # group by cv set
    for row_name, cv_df in preddf.groupby('cv_idx'):
        cv_df_m = cv_df.groupby('pred_thold').mean().sort('fp')
        ax.plot(cv_df_m['fp'], cv_df_m['tp'] )
    

    fname = infile[0].split('-')[0]
    ax.set_title(fname)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    f.savefig(outfile)

# @collate("sparkdata/*.samples.organized",
#          regex(r"data/(.+)-(.+).(\d\d)(.cv.*).samples"),
#          [td(r"\1-\2\4.predlinks"), td(r"\1-\2\4.assign")])
# def cv_collate_predlinks_assign(infiles_samples, (predlinks_outfile,
#                                                assign_outfile)):
    
    


@transform(cv_collate_predlinks_assign, suffix("predlinks.pickle"), "ari.pdf")
def plot_ari(infile, outfile):
    assigndf = pickle.load(open(infile[1], 'r'))['df']
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

    pipeline_run([create_data_latent,
                  spark_run_experiments,
                  get_samples,
                  samples_organize,
                  get_cvdata,
                  cvdata_organize,
                  cv_collate_predlinks_assign,
                  # create_inits,
                  # get_results,
                  # cv_collate, 
                  
                  plot_predlinks_roc,
                  plot_ari, 
                  # plot_circos_latent,
              ])
    
    
