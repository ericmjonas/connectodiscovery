import numpy as np
from nose.tools import *
from matplotlib import pylab
import model
import irm


def test_bins():
    bins = np.linspace(-20, 20, 100)

    mu = 3.0
    std = 2.0
    p = model.vec_gauss_prob(bins, mu, std)

    assert_almost_equal(np.sum(p), 1.0, 4)

    samps = np.random.normal(mu, std, size=1000000)
    h, _ = np.histogram(samps, bins)
    h = h.astype(float) / np.sum(h)
    assert irm.util.kl(h, p) < 0.001

    
    
def test_compute_mm_probs():
    bins = np.linspace(-14, 14, 100)

    params = [(0.5, 3.0, 1.0),
              (0.3, -2.0, 0.5), 
              (0.2, 0, 3.0)]
    N = 1000000
    i = np.random.multinomial(N, [p[0] for p in params])
    samps = np.zeros(N, dtype=np.float32)
    pos_i = 0
    for pos, count in enumerate(i):
        samps[pos_i:pos_i+count] = np.random.normal(params[pos][1], 
                                                   params[pos][2], size=count)
        pos_i += count
    samps = np.random.permutation(samps)

    h, _ = np.histogram(samps, bins)
    h = h.astype(float) / np.sum(h)

    p = model.compute_mm_probs(bins, params)
    assert_almost_equal(np.sum(p), 1.0, 4)

    assert irm.util.kl(h, p) < 0.001
    


