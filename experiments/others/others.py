import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from ruffus import * 
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
vblpcm = importr("VBLPCM")
network = importr("network")


import numpy as np
from matplotlib import pylab
import cPickle as pickle
import pandas

import irm

robjects.r.source("vblpcmroc2.R")

def vblpcm_cluster(adj_mat, G, d, edgecovs = None, STEPS=50, 
                   maxiter=1000):

    """
    Variational Bayes LPCM clustering with G groups and d latent dimensions. 
    no side attributes

    returns assignment vector. 

    hilariously, NOTHING apparently happens with unconnected vertices. 


    """

    assert adj_mat.dtype == np.float
    print "creating network"
    x_network = network.network(adj_mat)

    if edgecovs == None:
        print "starting"
        v_start = vblpcm.vblpcmstart(x_network, G=G, d=d)
    else:
        assert edgecovs.shape == adj_mat.shape
        assert edgecovs.dtype == np.float

        v_start = vblpcm.vblpcmstart(x_network, 
                                     G=G, d=d, edgecovs=edgecovs)
    print "Fitting"
    v_fit = vblpcm.vblpcmfit(v_start, STEPS=STEPS, maxiter=maxiter)
    print v_fit

    print "done"
    r_v_lambda = v_fit.rx2('V_lambda')
    
    roc = robjects.r.vblpcmroc2(v_fit)
    conv = v_fit.rx2('conv')[0]

    # well this is fun 
    robjects.r.assign("x_network", x_network)
    robjects.r.assign("v_start", v_start)
    robjects.r.assign("adj_mat", adj_mat)
    robjects.r.assign("v_fit", v_fit)
    robjects.r.assign("roc", roc)
                    
    robjects.r.save("x_network", "v_start", "adj_mat", "v_fit", "roc",
                    file=("rdata.%d.%d.rda" % (G, d)))
    
    #v_lambda= np.array(r_v_lambda)
    v_lambda = np.asarray(r_v_lambda)

    v_lambda = np.squeeze(v_lambda)

    assignments = []
    for i in range(len(adj_mat)):
        assignments.append(np.argmax(v_lambda[:, i]))
    assignments = np.array(assignments)
    return {'assignments' : assignments, 
            'roc' : float(roc[0]), 
            'conv' : conv}

