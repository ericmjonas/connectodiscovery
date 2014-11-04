"""
SPARK_DRIVER_MEMORY=4g ~/projects/sparktest/src/spark-1.1.0-bin-cdh4/bin/spark-submit  --conf spark.kryoserializer.buffer.mb=512 --conf spark.akka.frameSize=1000  --conf spark.executor.memory=4g --conf spark.python.worker.memory=4g --master local[2]

"""
import sys
# total hack, I should really know better
sys.path.append("../../code")

from ruffus import *
import cPickle as pickle
import numpy as np
import copy
import os, glob, sys, shutil
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


# def saveAsPickleFile(obj, tgtdir, overwrite=True):
#     try:
#         if overwrite:
#             shutil.rmtree(tgtdir)
#     except OSError:
#         pass
    
#     obj.saveAsPickleFile(tgtdir)

def s3n_delete(url):
    conn = boto.connect_s3()

    bucket_name = url[6:].split("/")[0]
    bucket = conn.get_bucket(bucket_name)
    
    path = url[6 + 1 + len(bucket_name):]
               
    delete_key_list = []
    for key in bucket.list(prefix=path):
        delete_key_list.append(key)
        if len(delete_key_list) > 100:
            bucket.delete_keys(delete_key_list)
            delete_key_list = []

    if len(delete_key_list) > 0:
        bucket.delete_keys(delete_key_list)        
    
def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


def to_f32(x):
    a = np.array(x).astype(np.float32)
    assert type(a) == np.ndarray
    return a

DEFAULT_CORES = 8
DEFAULT_RELATION = "ParRelation"
WORKING_DIR = "sparkdata"
S3_BUCKET = "jonas-testbucket2"
S3_PATH= "netmotifs/paper/experiments/synthdifferent"


def s3n_url(f):
    
    url =  "s3n://" + os.path.join(S3_BUCKET, S3_PATH, f)
    print url
    return url

    
def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))

EXPERIMENTS = [
    #('srm', 'cv_nfold_2', 'debug_2_100', 'debug_20'), 
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
    'debug_200' : {'ITERS' : 200, 
                  'kernels': irm.runner.default_kernel_anneal(1.0, 160)
              }
    }


# 

@files(None, td("srm.sourcedata"))
def create_data_srm(infile, outfile):
    np.random.seed(0)
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


@files(None, td("sbmnodist.sourcedata"))
def create_data_sbm_nodist(infile, outfile):
    np.random.seed(0)
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



@files(None, td("lpcm.sourcedata"))
def create_data_latent_position(infile, outfile):
    np.random.seed(0)

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


@files(None, td("mm.sourcedata"))
def create_data_mixed_membership(infile, outfile):
    """
    Each cell can have some subset of 4 properties, and there exist
    arbitrary predicates that dictate whether or not it connects
    to other cells. 

    This is really equivalent to just 2^3 possible latent states, 
    but with a simpler structure on connectivity
    """
    
    np.random.seed(0)
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


def create_cv_pure(data, meta, 
                   cv_i, cv_config_name, cv_config):
    """ 
    Creates a single cross-validated data set 
    """

    shape = data['relations']['R1']['data'].shape
    N =  shape[0] * shape[1]
    if cv_config['type'] == 'nfold':
        np.random.seed(0) # set the seed

        perm = np.random.permutation(N)
        subset_size = N / cv_config['N']
        subset = perm[cv_i * subset_size:(cv_i+1)*subset_size]
        
        observed = np.ones(N, dtype=np.uint8)
        observed[subset] = 0
        data['relations']['R1']['observed'] = np.reshape(observed, shape)

    else:
        raise Exception("Unknown cv type")
    
    

    meta = copy.deepcopy(meta)
    meta['cv'] = {'cv_i' : cv_i,
                  'cv_config_name' : cv_config_name}
    
    return (data, meta)


def create_init_pure(irm_latent, irm_data, OUT_N, 
                init= None, keep_ground_truth=True):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)

    # FIXME : add ability to init multiple domains
    """
    irm_latents = []

    rng = irm.RNG()

    irm_model = irm.irmio.create_model_from_data(irm_data, rng=rng)
    for c in range(OUT_N):
        np.random.seed(c)

        latent = copy.deepcopy(irm_latent)

        d_N = len(latent['domains']['d1']['assignment'])
        if init['type'] == 'fixed':
            group_num = init['group_num']

            a = np.arange(d_N) % group_num
            a = np.random.permutation(a)

        elif init['type'] == 'crp':
            alpha = init['alpha']
            a = irm.util.crp_draw(d_N, alpha)
            a = np.random.permutation(a) 
        elif init['type'] == 'truth':
            a = latent['domains']['d1']['assignment']
            
        else:
            raise NotImplementedError("Unknown init type")
            
        if (not keep_ground_truth) or (c > 0) : # first one stays the same
            latent['domains']['d1']['assignment'] = a

        # generate new suffstats, recompute suffstats in light of new assignment

        irm.irmio.set_model_latent(irm_model, latent, rng)

        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=2)

        yield irm.irmio.get_latent(irm_model)

    
def experiment_generator():
    for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
        data_filename = get_dataset(data_name)[0]
        print "data filename =", data_filename
        df = "%s-%s-%s-%s" % (data_name, cv_config_name, init_config_name, kernel_config_name)
        
        out_files = [td(df + x) for x in [ ".samples",
                                       ".cvdata", ".inits"]]
        
        yield data_filename, out_files, cv_config_name, init_config_name, kernel_config_name

@follows(create_data_latent)
@files(experiment_generator)
def spark_run_experiments(data_filename, (out_samples, out_cv_data, out_inits), 
                          cv_config_name, init_config_name, kernel_config_name):

    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"
    meta_filename = basename + ".meta"
    

    data = pickle.load(open(data_filename))
    true_latent = pickle.load(open(latent_filename))
    meta = pickle.load(open(meta_filename))
    
    sc = SparkContext(batchSize=1)
    
    data_broadcast = sc.broadcast(data)
    cv_config = CV_CONFIGS[cv_config_name]
    CV_N = cv_config['N']
    init = INIT_CONFIGS[init_config_name]
    INIT_N = init['N']
    print "WE ARE RUNNING THIS THING" 
    def cv(cv_i):
        """ 
        returns key, (data, meta)
        """
        cv_key = "%s.%d" % (cv_config_name, cv_i)
        return cv_key, create_cv_pure(data, #_broadcast.value, 
                                      meta, cv_i,
                                      cv_config_name, cv_config)

    
    cv_data_rdd = sc.parallelize(range(CV_N), 
                                 CV_N).map(cv).cache()
    
    def create_init_flat((key, (data, meta))):
        """
        returns latent
        """
        res = []
        for latent_i, latent in enumerate( create_init_pure(true_latent, data, INIT_N, init['config'])):
            yield "%s.%s" % (key, latent_i), (data, meta, latent)


    init_latents_rdd  = cv_data_rdd.flatMap(create_init_flat)

    def inference((data, meta, init)):

        return run_exp_pure(data, init, kernel_config_name, 0)
       # FIXME do we care about this seed? 
        
    joined = init_latents_rdd.repartition(CV_N*INIT_N).cache()
    print "THERE ARE", joined.getNumPartitions(), "partitions" 
    results = joined.mapValues(inference)


    for rdd, name in [(results, out_samples),
                      (init_latents_rdd, out_inits),
                      (cv_data_rdd, out_cv_data)]:
        url = s3n_url(name)
        s3n_delete(url)
        rdd.saveAsPickleFile(url)
        pickle.dump({'url' : url}, open(name, 'w'))
    
    sc.stop()
    
def inference_run(data, latent, 
                  kernel_config, 
                  ITERS, seed, VOLUME_NAME, init_type=None, 
                  fixed_k = False, 
                  latent_samp_freq=20, 
                  relation_class = "Relation", 
                  cores = 1):


    if relation_class == "Relation":
        relation_class = irm.Relation
    elif relation_class == "ParRelation":
        relation_class = irm.ParRelation
    else:
        raise NotImplementedError("unknown relation class %s" % relation_class)

    if cores == 1:
        threadpool = None
    else:
        print "Creating threadpool with", cores, "cores"
        threadpool = irm.pyirm.ThreadPool(cores)

    chain_runner = irm.runner.Runner(latent, data, kernel_config, seed, 
                                     fixed_k = fixed_k, 
                                     relation_class = relation_class,
                                     threadpool = threadpool)

    if init_type != None:
        chain_runner.init(init_type)

    scores = []
    times = []
    latents = {}
    def logger(iter, model, res_data):
        print "Iter", iter
        scores.append(model.total_score())
        times.append(time.time())

        if iter % latent_samp_freq == 0:
            latents[iter] = chain_runner.get_state(include_ss=False)
    chain_runner.run_iters(ITERS, logger)
        
    return scores, chain_runner.get_state(), times, latents


def run_exp_pure(data, init, kernel_config_name, seed):
    # put the filenames in the data

    kc = KERNEL_CONFIGS[kernel_config_name]
    ITERS = kc['ITERS']
    kernel_config = kc['kernels']
    fixed_k = kc.get('fixed_k', False)
    cores = kc.get('cores', DEFAULT_CORES)
    relation_class = kc.get('relation_class', DEFAULT_RELATION)


    res = inference_run(data, init,
                        kernel_config, 
                        ITERS,
                        seed,
                        fixed_k,
                        relation_class=relation_class,
                        cores = cores)
    


    
    return {
            'res' : res, 
            'kernel_config_name' : kernel_config_name}



@transform(spark_run_experiments, suffix('.samples'), '.samples.pickle')
def get_samples((exp_samples, exp_cvdata, exp_inits), out_filename):
    sample_metadata = pickle.load(open(exp_samples, 'r'))
    
    sc = SparkContext()
    results_rdd = sc.pickleFile(sample_metadata['url'])

    save_rdd_elements(results_rdd, out_filename)
    
    sc.stop()

@transform(spark_run_experiments, suffix('.samples'), '.cvdata.pickle')
def get_cvdata((exp_samples, exp_cvdata, exp_inits), out_filename):
    sample_metadata = pickle.load(open(exp_samples, 'r'))
    
    sc = SparkContext()
    results_rdd = sc.pickleFile(sample_metadata['url'])
    pickle.dump(results_rdd.collect(),
                open(out_filename, 'w'))
    sc.stop()

def save_rdd_elements(rdd, filename_base):
    """
    save each element of the rdd in filename_base + .nnn on the 
    local disk

    FIXME: This is really just a staging area to get around
    instantiating the entire rdd in memory

    """
    conn = boto.connect_s3()
    bucket = conn.get_bucket(S3_BUCKET)
    key_name_base = filename_base

    def create_key_name(index):
        return S3_PATH + key_name_base + (".%08d" %  index)

    def save_s3((obj, index)):
        a = pickle.dumps(obj)
        k = boto.s3.key.Key(bucket)
        k.key = create_key_name(index)
        k.set_contents_from_string(a)
    

    # materialize, save
    SIZE_OF_RDD = rdd.count()
    rdd.zipWithIndex().foreach(save_s3)


    # now redownload
    outfiles = []
    for i in range(SIZE_OF_RDD):
        key = bucket.new_key(create_key_name(i))
        contents = key.get_contents_as_string()
        filename = filename_base + (".%03d" % i)
        outfiles.append(filename)
        
        key.get_contents_to_filename(filename)
    
    pickle.dump(outfiles, 
                open(filename_base, 'w'))


@follows(get_samples)
@collate("sparkdata/*.cv.*.samples",
         regex(r"data/(.+)-(.+).(\d\d)(.cv.*).samples"),
         [td(r"\1-\2\4.predlinks"), td(r"\1-\2\4.assign")])
def cv_collate_predlinks_assign(infiles_samples, (predlinks_outfile,
                                               assign_outfile)):
    """
    aggregate across samples and cross-validations
    """
    s = infiles_samples[0].split('-')

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

    for sample_name in infiles_samples:
        # this loop runs once per cross-validated dataset
        
        samples = pickle.load(open(sample_name, 'r'))
        
        N = len(data_conn)

        # GET THE CROSSVALIDATED DATA
        s = sample_name.split('-')

        cv_name = s[0] + "-" + s[1]
        cv_data = pickle.load(open(cv_name, 'r'))

        # get the cv idx for later use
        cv_idx = int(s[1][-10:-8])
        
        # FIGURE OUT WHICH ENTRIES WERE MISSING
        heldout_idx = np.argwhere((cv_data['relations']['R1']['observed'].flatten() == 0)).flatten()
        heldout_true_vals = truth_mat.flatten()[heldout_idx]
        
        heldout_true_vals_t_idx = np.argwhere(heldout_true_vals > 0).flatten()
        heldout_true_vals_f_idx = np.argwhere(heldout_true_vals == 0).flatten()
        chain_n = 0

        for chain_i, chain in enumerate(samples['chains']):
            if type(chain['scores']) != int:
                irm_latent_samp = chain['state']

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
                                            'score' : chain['scores'][-1], 
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
                                            'score' : chain['scores'][-1],
                                            'cv_idx' : cv_idx, 
                                            'true_assign' : true_assign,
                                            'assign' : irm_latent_samp['domains']['d1']['assignment']})
                                            


    predlinks_df = pandas.DataFrame(predlinks_outputs)
    pickle.dump({'df' : predlinks_df}, 
                 open(predlinks_outfile, 'w'))

    a_df = pandas.DataFrame(assignments_outputs)
    pickle.dump({'df' : a_df}, 
                 open(assign_outfile, 'w'))

    
@transform(cv_collate_predlinks_assign, suffix(".predlinks"), ".roc.pdf")
def plot_predlinks_roc(infile, outfile):
    preddf = pickle.load(open(infile[0], 'r'))['df']
    preddf['tp'] = preddf['t_t'] / preddf['t_tot']
    preddf['fp'] = preddf['f_t'] / preddf['f_tot']
    preddf['frac_wrong'] = 1.0 - (preddf['t_t'] + preddf['f_f']) / (preddf['t_tot'] + preddf['f_tot'])

    f = pylab.figure(figsize=(4, 4))
    ax = f.add_subplot(1, 1, 1)
    
    # group by cv set
    for row_name, cv_df in preddf.groupby('cv_idx'):
        print "PLOTTING", row_name
        cv_df = cv_df.sort('fp')
        ax.plot(cv_df['fp'], cv_df['tp'], c='k', alpha=0.5)

    fname = infile[0].split('-')[0]
    ax.set_title(fname)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    f.savefig(outfile)



@transform(cv_collate_predlinks_assign, suffix(".predlinks"), ".ari.pdf")
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

@follows(get_samples)
@transform(get_samples, suffix(".samples.pickle"), ".samples.organized.pickle")
def samples_organize(infile, outfile):
    associated_files_list = pickle.load(open(infile, 'r'))
    for f in associated_files_list:
        a = pickle.load(open(f, 'r'))
        print infile, a[0]
        
        
if __name__ == "__main__":
    pipeline_run([create_data_latent,
                  spark_run_experiments,
                  get_samples,
                  samples_organize,
                  get_cvdata
                  # create_inits,
                  # get_results,
                  # cv_collate, 
                  # cv_collate_predlinks_assign,
                  # plot_predlinks_roc,
                  # plot_ari, 
                  # plot_circos_latent
              ])
    
    
