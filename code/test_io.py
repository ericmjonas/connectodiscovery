import numpy as np
from nose.tools import * 
from matplotlib import pylab

import connattribio
import irm

def test_io():
    N = 10
    desc = {'f1' : {'data' : np.zeros(N, dtype=np.bool), 
                    'model' : 'BetaBernoulli'}}
    
    latent, data = connattribio.create_mm(desc)

    latent, data = irm.data.synth.prior_generate(latent, data)
    print data
    print latent

    assert_equal(len(latent['domains']), 2)
    assert_equal(len(latent['relations']), 1)
    assert_equal(len(data['domains']), 2)
    assert_equal(len(data['relations']), 1)
    
    
    rng = irm.RNG()
    irm_model = irm.irmio.create_model_from_data(data, rng=rng)
    irm.irmio.set_model_latent(irm_model, latent, rng)

def test_mixture():
    N = 100

    np.random.seed(0)

    d = np.zeros(N, dtype=np.float32)
    for i in range(N/2):
        d[i] = np.random.normal(-4, 1)
        d[i+N/2] = np.random.normal(4, 1)

    d = np.random.permutation(d)

    desc = {'f1' : {'data' : d, 
                    'model' : 'NormalInverseChiSq'}}

    latent, data = connattribio.create_mm(desc)
    latent['domains']['d1']['assignment'] = np.arange(N) % 10


    latent, data = irm.data.synth.prior_generate(latent, data)
    
    
    rng = irm.RNG()
    irm_model = irm.irmio.create_model_from_data(data, rng=rng)
    irm.irmio.set_model_latent(irm_model, latent, rng)
    
    kernel_config = irm.runner.default_kernel_anneal()
    
    for i in range(200):
        irm.runner.do_inference(irm_model, rng, kernel_config, i)

        new_latent = irm.irmio.get_latent(irm_model)
        a = new_latent['domains']['d1']['assignment']
        
        print irm.util.assign_to_counts(a)
        print new_latent['relations']['r_f1']['hps']

def test_mixture_bb():
    ENTITY_PER_GROUP = 50
    GROUPS = 4
    N = ENTITY_PER_GROUP * GROUPS
    DIM = 4 

    np.random.seed(0)


    gv = np.random.beta(0.2, 0.2, size=(GROUPS, DIM))


    mat = np.zeros((N, DIM), dtype=np.uint8)
    for g in range(GROUPS):
        for i in range(ENTITY_PER_GROUP):
            for d in range(DIM):
                mat[g * ENTITY_PER_GROUP + i, d] = np.random.rand() < gv[g, d] 


    #mat = np.random.permutation(mat)
    desc = {}
    for d in range(DIM):
        desc['f%d' % d] = {'data' : mat[:, d], 
                           'model' : 'BetaBernoulli'}

    latent, data = connattribio.create_mm(desc)
    latent['domains']['d1']['assignment'] = np.arange(N) % 10


    latent, data = irm.data.synth.prior_generate(latent, data)
    
    
    rng = irm.RNG()
    irm_model = irm.irmio.create_model_from_data(data, rng=rng)
    irm.irmio.set_model_latent(irm_model, latent, rng)
    
    kernel_config = irm.runner.default_kernel_anneal()


    for i in range(150):
        irm.runner.do_inference(irm_model, rng, kernel_config, i)

        new_latent = irm.irmio.get_latent(irm_model)
        a = new_latent['domains']['d1']['assignment']
        
        print irm.util.assign_to_counts(a)
        print new_latent['relations']['r_f1']['hps']

    pylab.imshow(mat[np.argsort(a)])
    pylab.show()
                 
