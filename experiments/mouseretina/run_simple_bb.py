"""
The purpose of this is just to run the dang model on the beta-bernoulli
data
"""

import numpy as np
import cPickle as pickle
import irm
import copy
import time



THOLD_IDX = 1

# load in the retina data
rd = pickle.load(open("data/retina.%d.data.pickle" % THOLD_IDX, 'r'))

conn_mat = rd['conn_mat']
dist_mats = rd['dist_mats']

irm_latent, irm_data = irm.irmio.default_graph_init(conn_mat, "BetaBernoulliNonConj")


# set the hypers

irm_latent['relations']['R1']['hps'] = {'alpha' : 1.0, 
                                        'beta' : 1.0}

basename = "data/retina.%d.simple_bb.data" % THOLD_IDX
pickle.dump(irm_data, open(basename, 'w'))

kernel_config = pickle.load(open('anneal_slow_1000.config', 'r'))


seed = 0

chain_runner = irm.runner.Runner(irm_latent, irm_data,
                                 kernel_config, seed)
ITERS = 1000

latent_samp_freq = 50
scores = []
times = []
latents = {}
def logger(iter, model, logger2):
    print "Iter", iter
    scores.append(model.total_score())
    times.append(time.time())
    if iter % latent_samp_freq == 0:
        latents[iter] = chain_runner.get_state(include_ss=False)
chain_runner.run_iters(ITERS, logger)
state = chain_runner.get_state()

chains = []
        
chains.append({'scores' : scores, 
               'state' : state, 
               'times' : times, 
               'latents' : latents})
        
outfilename = "%s-fixed_4-anneal_slow_400.samples" % basename

pickle.dump({'chains' : chains, 
             'exp' : None}, 
            open(outfilename, 'w'))


# run it for bb-nonconj


# call it good
