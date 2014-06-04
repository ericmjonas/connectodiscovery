import numpy as np
from nose.tools import *
from matplotlib import pylab
import model
import irm
import mixmodel

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
                                                    np.sqrt(params[pos][2]), 
                                                    size=count)
        pos_i += count
    samps = np.random.permutation(samps)

    h, _ = np.histogram(samps, bins)
    h = h.astype(float) / np.sum(h)

    p = model.compute_mm_probs(bins, params)
    assert_almost_equal(np.sum(p), 1.0, 4)

    assert irm.util.kl(h, p) < 0.001
    


def test_mm_mixmodel():
    np.random.seed(0)

    import distributions
    N = 100

    data = np.zeros(N, dtype=np.float32)
    MODEL = model.NonConjGaussian(0.5)

    data[::2] = np.random.normal(-8, 0.5, N/2)
    data[1::2] = np.random.normal(8, 0.5, N/2)
    f = mixmodel.Feature(data, MODEL)

    mm = mixmodel.MixtureModel(N, {'f1' : f})
    
    rng = None
    # random init
    grp = {}
    for i, g in enumerate(np.random.permutation(np.arange(N) % 10)):
        if g not in grp:
            grp[g] = mm.create_group(rng)
        mm.add_entity_to_group(grp[g], i)
    print mm.score()

    for i in range(1000):
        irm.gibbs.gibbs_sample_type_nonconj(mm, 10, rng)
        print mm.score()

    assert_equal(irm.util.count(mm.get_assignments()).values(), [50, 50])


def test_data_prob_mm():
    """
    Testing this function is a challenge. Can we empirically calculate via 
    forward-sampling the 2-d histogram and then analytically calculate for a bunch
    of test points? 
    
    """
    mu_vect = [-2.0, 0.0, 3.0]
    sigmasq_vect = [1.0, 0.2, 0.4]
    pi_vect = [0.25, 0.4, 0.35]
    
    params = zip(pi_vect, mu_vect, sigmasq_vect)
    D = 1
    N = 100000
    data = np.zeros((N, D))
    for n in range(N):
        data[n] = model.sample_from_mm(D, params)

    GRID_POINTS = 100
    grid = np.linspace(-5, 5, GRID_POINTS)
    grid.shape = (GRID_POINTS, 1)
    scores = np.zeros(GRID_POINTS)
    for g in range(GRID_POINTS):
        scores[g] = model.data_prob_mm(np.array(grid[g]), 
                                       mu_vect, sigmasq_vect, pi_vect)

    print data.shape, grid.shape

    h, _ = np.histogram(data[:, 0], bins=grid[:, 0])
    h = h.astype(float) / np.sum(h)
    print h
    print data[:, 0]
    print grid[:, 0]
    grid_width = grid[1] - grid[0]
    pylab.plot(grid, np.exp(scores), c='g')
    pylab.plot(grid[:-1] + grid_width/2., h/grid_width)

    pylab.show()
