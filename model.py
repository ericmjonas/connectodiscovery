import numpy as np
from scipy.special import erf, gammaln
import irm

"""
Test mixture model of things-that-look-like-distributions
"""


def vec_gauss_prob(bins, mu, std):
    """
    Probability mass of N(mu, std^2) in each bin

    """
    
    a = erf((bins[1:]-mu)/(std*np.sqrt(2.0))) - erf((bins[:-1]-mu)/(std*np.sqrt(2.0)))
    return a/2.0
    
def compute_mm_probs(bins, params):
    """
    params are [(pi, mu, sigmasq)]
    
    returns len(bins)-1 bins
    """
    N = len(bins) - 1
    K = len(params)
    out = np.zeros(N, dtype=np.float32)
    for pi, mu, sigma in params:
        out += pi * vec_gauss_prob(bins, mu, sigma)

    return out

fln_table = [0., 0., .69314, 1.79175, 3.17805, 4.78749,
    6.57925, 8.52516, 10.60460, 12.80182]

def factorialln(x):
    if x < 10:
        return fln_table[x]
    else:
        return gammaln(x + 1)

def log_multinom_dens(x, p):
    p_eps = 1e-10
    l = np.dot(x, np.log(p + p_eps))
    # normalizing factor

    Z = factorialln(np.sum(x)) - np.sum(gammaln(x + 1))
    return l + Z


class NonConjGaussian(object):
    """
    Simple model of nonconjugate gaussians, const var
    
    """
    
    def __init__(self, var):
        self.var = var

    def create_hp(self):
        return None

    def add_data(self, ss, val):
        ss['score'] += irm.util.log_norm_dens(val, ss['mu'], self.var)
        
    def remove_data(self, ss, val):
        ss['score'] -= irm.util.log_norm_dens(val, ss['mu'], self.var)
        
    def create_ss(self):
        return {'mu' : np.random.normal(0, 10), 
                'score' : 0.0}
        
    def pred_prob(self, hps, ss, val):
        mu = ss['mu']
        return irm.util.log_norm_dens(val, mu, self.var)
        
    def data_prob(self, hps, ss, assign):
        return ss['score']


class BinnedDist(object):
    """
    Simple model of nonconjugate gaussians, const var
    
    """
    
    def __init__(self, BIN_N, COMP_K, comp_std = 1.0):
        self.BIN_N = BIN_N
        self.COMP_K = COMP_K
        self.comp_std = comp_std
        self.bins = np.linspace(0, 1.0, self.BIN_N+1)
    def create_hp(self):
        return None

    def add_data(self, ss, val):
        pass
    def remove_data(self, ss, val):
        pass

    def compute_mm_probs(self, pis, mus, vars):
        return compute_mm_probs(self.bins, 
                                zip(pis, mus, vars))

    def create_ss(self):
        ss = {'mu' : np.random.rand(self.COMP_K), 
              'var' : np.ones(self.COMP_K)*self.comp_std**2, 
              'pi' : np.random.dirichlet(np.ones(self.COMP_K))}
        return ss

    def pred_prob(self, hps, ss, val):
        """
        Val is a list of counts
        """
        p =  self.compute_mm_probs(ss['pi'], ss['mu'], ss['var'])
        return log_multinom_dens(val, p)
        
    def data_prob(self, hps, ss, data):
        score = 0.0 
        for val in data:
            score += self.pred_prob(hps, ss, val)
        return score

