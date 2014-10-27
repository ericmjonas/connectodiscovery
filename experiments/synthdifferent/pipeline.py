"""
"""
import sys
# total hack, I should really know better
sys.path.append("../../code")

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
import multyvac

import irm
import irm.data
import util
from irm import rand
import predutil



def dist(a, b):
    return np.sqrt(np.sum((b-a)**2))


def to_f32(x):
    a = np.array(x).astype(np.float32)
    assert type(a) == np.ndarray
    return a


VOLUME_NAME = "connattrib_mouseretina_paper_synthdiff"
MULTYVAC_LAYER = "test11"
DEFAULT_CORES = 8
DEFAULT_RELATION = "ParRelation"
WORKING_DIR = "data"


@files(None, "volume.%s.sentinel" % VOLUME_NAME)
def create_volume(infile, outfile):
    multyvac.volume.create(VOLUME_NAME, '/%s' % VOLUME_NAME)
    vol = multyvac.volume.get(VOLUME_NAME) 
    fname = "%s/dir.setinel" % WORKING_DIR
    fid = file(fname, 'w')
    fid.write("test")
    fid.close()
    vol.put_file(fname, fname)
   
    open(outfile, 'w').write("done\n")

    
def td(fname): # "to directory"
    return os.path.join(WORKING_DIR, fname)

def get_dataset(data_name):
    return glob.glob(td("%s.data" %  data_name))

EXPERIMENTS = [
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
                               'type' : 'nfold'}}


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
    'debug_20' : {'ITERS' : 20, 
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


def cv_generator():
    for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset(data_name):
            name, _ = os.path.splitext(data_filename)
            for cv_i in range(CV_CONFIGS[cv_config_name]['N']):
                cv_name_base = "%s-%s.%02d.cv" % (name, cv_config_name, cv_i)
                
                yield data_filename, [cv_name_base + ".data",
                                      cv_name_base + ".latent",
                                      cv_name_base + ".meta"], cv_i, cv_config_name, CV_CONFIGS[cv_config_name]


@follows(create_data_latent)            
@files(cv_generator)
def create_cv(data_filename, (out_data_filename,
                              out_latent_filename,
                              out_meta_filename), cv_i, cv_config_name, cv_config):
    """ 
    Creates a single cross-validated data set 
    """
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"
    meta_filename = basename + ".meta"


    data = pickle.load(open(data_filename))
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
    
    pickle.dump(data, open(out_data_filename, 'w'))
                    
    latent = pickle.load(open(latent_filename))
    pickle.dump(latent, open(out_latent_filename, 'w'))
                    
    meta = pickle.load(open(meta_filename))
    meta['cv'] = {'cv_i' : cv_i,
                  'cv_config_name' : cv_config_name}
    
    pickle.dump(meta, open(out_meta_filename, 'w'))
                    
def init_generator():
    for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset("%s-*%s*" % (data_name, cv_config_name)):
            name, _ = os.path.splitext(data_filename)

            # now generate one of these for every single cv dataset:
            yield data_filename, ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])], init_config_name, INIT_CONFIGS[init_config_name]


@follows(create_data_latent)            
@follows(create_cv)            
@files(init_generator)
def create_inits(data_filename, out_filenames, init_config_name, init_config):
    basename, _ = os.path.splitext(data_filename)
    latent_filename = basename + ".latent"
    
    irm.experiments.create_init(latent_filename, data_filename, 
                                out_filenames, 
                                init= init_config['config'], 
                                keep_ground_truth=False)



    
def experiment_generator():
    for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
        for data_filename in get_dataset("%s-*%s*" % (data_name, cv_config_name)):
            name, _ = os.path.splitext(data_filename)

            inits = ["%s-%s.%02d.init" % (name, init_config_name, i) for i in range(INIT_CONFIGS[init_config_name]['N'])]
            
            exp_name = "%s-%s-%s.wait" % (data_filename, init_config_name, kernel_config_name)
            yield [data_filename, inits], exp_name, kernel_config_name

            
@follows(create_volume)
@follows(create_inits)
@files(experiment_generator)
def run_exp((data_filename, inits), wait_file, kernel_config_name):
    # put the filenames in the data
    irm.experiments.to_bucket(data_filename, VOLUME_NAME)
    test = irm.experiments.from_bucket(data_filename, VOLUME_NAME)

    [irm.experiments.to_bucket(init_f, VOLUME_NAME) for init_f in inits]
    kernel_config_filename = kernel_config_name + ".pickle"

    kc = KERNEL_CONFIGS[kernel_config_name]
    ITERS = kc['ITERS']
    kernel_config = kc['kernels']
    fixed_k = kc.get('fixed_k', False)
    cores = kc.get('cores', DEFAULT_CORES)
    relation_class = kc.get('relation_class', DEFAULT_RELATION)

    pickle.dump(kernel_config, open(kernel_config_filename, 'w'))

    irm.experiments.to_bucket(kernel_config_filename, VOLUME_NAME)


    CHAINS_TO_RUN = len(inits)

    
    jids = []

    for init_i, init in enumerate(inits):
        jid = multyvac.submit(irm.experiments.inference_run, 
                              init, 
                              data_filename, 
                              kernel_config_filename, 
                              ITERS, 
                              init_i, 
                              VOLUME_NAME, 
                              None, 
                              fixed_k, 
                              relation_class = relation_class, 
                              cores = cores, 
                              _name="%s-%s-%s" % (data_filename, init, 
                                                  kernel_config_name), 
                              _layer = MULTYVAC_LAYER,
                              _multicore = cores, 
                              _core = 'f2')
        jids.append(jid)


    pickle.dump({'jids' : jids, 
                'data_filename' : data_filename, 
                'inits' : inits, 
                'kernel_config_name' : kernel_config_name}, 
                open(wait_file, 'w'))



@transform(run_exp, suffix('.wait'), '.samples')
def get_results(exp_wait, exp_results):
    d = pickle.load(open(exp_wait, 'r'))
    
    chains = []
    # reorg on a per-seed basis
    for jid in d['jids']:
        job = multyvac.get(jid)
        job.wait()
        
        print "getting", jid, job.status
        if job.status == 'done':
            chain_data = job.get_result()

            chains.append({'scores' : chain_data[0], 
                           'state' : chain_data[1], 
                           'times' : chain_data[2], 
                           'latents' : chain_data[3]})
        
    pickle.dump({'chains' : chains, 
                 'exp' : d}, 
                open(exp_results, 'w'))



CIRCOS_DIST_THRESHOLDS = [10]

@transform(get_results, suffix(".samples"), 
           [(".circos.%02d.png" % d,)  for d in range(len(CIRCOS_DIST_THRESHOLDS))])
def plot_circos_latent(exp_results, 
                       out_filenames):

    sample_d = pickle.load(open(exp_results))
    chains = sample_d['chains']
    
    exp = sample_d['exp']
    data_filename = exp['data_filename']
    data = pickle.load(open(data_filename))
    data_basename, _ = os.path.splitext(data_filename)
    meta = pickle.load(open(data_basename + ".meta"))
    print meta.keys()
    meta_infile = meta['infile']
    print "meta_infile=", meta_infile

    meta_file = pickle.load(open(meta_infile, 'r'))

    conn_and_dist = meta['conn_and_dist']


    chains = [c for c in chains if type(c['scores']) != int]
    CHAINN = len(chains)

    chains_sorted_order = np.argsort([d['scores'][-1] for d in chains])[::-1]
    chain_pos = 0

    best_chain_i = chains_sorted_order[chain_pos]
    best_chain = chains[best_chain_i]
    sample_latent = best_chain['state']
    cell_assignment = np.array(sample_latent['domains']['d1']['assignment'])

    ca = irm.util.canonicalize_assignment(cell_assignment)
    ca = np.argsort(ca)
    #print meta_file['nodes_with_meta']['class']
    #ca = meta_file['nodes_with_meta']['class']
    c = conn_and_dist['link'][ca]
    c = c[:, ca]
    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    irm.plot.plot_t1t1_latent(ax, conn_and_dist['link'],
                              cell_assignment)
    #ax.imshow(c, 
    #          interpolation='nearest', cmap=pylab.cm.Greys)

    f.savefig(out_filenames[0][0])

    
@follows(get_results)
@collate("data/*.cv.*.samples",
         regex(r"data/(.+)-(.+).(\d\d)(.cv.*).samples"),
         r"\1-\2\4.agg")

def cv_collate(infiles, outfile):
    print infiles
    print outfile
    open(outfile, 'w').write("none\n")
    

@follows(get_results)
@collate("data/*.cv.*.samples",
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


if __name__ == "__main__":
    pipeline_run([create_data_latent, create_inits,
                  get_results,
                  cv_collate, 
                  cv_collate_predlinks_assign,
                  plot_predlinks_roc,
                  plot_ari, 
                  plot_circos_latent
              ])
    
    
