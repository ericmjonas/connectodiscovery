import numpy as np
from nose.tools import *
from matplotlib import pylab
import mixmodel
import irm
import gibbs

def test_mm():
    np.random.seed(0)

    import distributions
    N = 100

    data = np.zeros(N, dtype=np.float32)
    MODEL = distributions.conjugate.nich
    data[::2] = -1
    data[1::2] = 1
    f = mixmodel.Feature(data, MODEL)
    f.hps = MODEL.HP(0.0, 1.0, 1.0, 1.0)

    mm = mixmodel.MixtureModel(N, {'f1' : f})
    
    rng = None
    # random init
    grp = {}
    for i, g in enumerate(np.random.permutation(np.arange(N) % 10)):
        if g not in grp:
            grp[g] = mm.create_group(rng)
        mm.add_entity_to_group(grp[g], i)
    print mm.score()

    for i in range(10):
        gibbs.gibbs_sample(mm, rng)
        print mm.score()

    assert_equal(irm.util.count(mm.get_assignments()).values(), [50, 50])
