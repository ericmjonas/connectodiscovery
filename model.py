import numpy as np
from scipy.special import erf, gammaln
import irm
import copy
from matplotlib import pylab
from scipy.stats import chi2, t, norm


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
    for pi, mu, sigmasq in params:
        out += pi * vec_gauss_prob(bins, mu, np.sqrt(sigmasq))

    return out

def sample_from_mm(N, params):
    """
    Generate N from the params
    params are [(pi, mu, sigmasq)]
    """
    
    pi_vect = [p[0] for p in params]
    clust_sizes = np.random.multinomial(N, pi_vect)
    d = np.zeros(N)
    pos = 0
    for ci, c_size in enumerate(clust_sizes):
        d[pos:pos+c_size] = np.random.normal(params[ci][1], 
                                             np.sqrt(params[ci][2]), 
                                             size=c_size)
        pos += c_size
    
    return np.random.permutation(d)

    

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

def log_dirichlet_dens(p, alphas):
    score = 0.0
    beta_alpha = 0.0
    for i in range(len(p)):
        score += (alphas[i] - 1.) * np.log(p[i])
        beta_alpha += gammaln(alphas[i])
    beta_alpha -= gammaln(np.sum(alphas))
    score -= beta_alpha
    return score
                          

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
    """
    
    def __init__(self, BIN_N, COMP_K, comp_var = 1.0):
        self.BIN_N = BIN_N
        self.COMP_K = COMP_K
        self.comp_var = comp_var
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
              'var' : np.ones(self.COMP_K)*self.comp_var**2, 
              'pi' : np.random.dirichlet(np.ones(self.COMP_K))}
        return ss

    def pred_prob(self, hps, ss, val):
        """
        Val is a list of counts
        """
        p =  self.compute_mm_probs(ss['pi'], ss['mu'], ss['var'])
        #p = p / np.sum(p)
        return log_multinom_dens(val, p)
        
    def data_prob(self, hps, ss, data):
        score = 0.0 
        for val in data:
            score += self.pred_prob(hps, ss, val)
        return score

class MMDist(object):
    """
    We assume that the observations are mostly in the range 0, 1

    Hypers: 
    comp_k : Number of gaussians [parameter]
    dir_alpha = alpha for the dirichlet prior on pi
    var_scale = 0.1
    """
    
    def __init__(self):
        self.CHI_VAL = 1.0
        
    def create_hp(self):
        return {'comp_k' : 4, 
                'dir_alpha' : 1.0, 
                'var_scale' : 0.1}

    def sample_from_prior(self, hps, rng):
        # uniform distribution on mu
        comp_k = hps['comp_k']
        
        mu = np.random.rand(comp_k) 
        var = chi2.rvs(self.CHI_VAL, size=comp_k)*hps['var_scale']
        pi = np.random.dirichlet(np.ones(comp_k)*hps['dir_alpha'])
        
        return mu, var, pi

    def ss_sample_new(self, hps, rng):
        mu, var, pi = self.sample_from_prior(hps, rng)
        return {'mu' : mu, 
                'var' : var, 
                'pi' : pi}
    
    def score_prior(self, ss, hps):
        score = 0
        epsilon = 0.001
        for mu, var in zip(ss['mu'], ss['var']):
            if mu <= epsilon or mu >= (1-epsilon):
                return -np.inf
            score += np.log(chi2.pdf(var/hps['var_scale'], self.CHI_VAL))
        
        score += log_dirichlet_dens(ss['pi'], np.ones(hps['comp_k'])*hps['dir_alpha'])
        return score
        
    def add_data(self, ss, val):
        pass

    def remove_data(self, ss, val):
        pass

    def create_ss(self, hps, rng):
        return self.ss_sample_new(hps, rng)


    def pred_prob(self, hps, ss, val):
        """
        Val is a list of observations
        """
        return data_prob_mm_fast(val, ss['mu'], ss['var'], ss['pi']) # /len(val)
        

        
    def score_likelihood(self, hps, ss, data):
        score = 0.0 
        for val in data:
            score += self.pred_prob(hps, ss, val)
        return score
    
    def data_prob(self, hps, ss, data):
        likelihood_score = self.score_likelihood(hps, ss, data)
        prior_score = self.score_prior(ss, hps)
        return likelihood_score + prior_score

# mh the binned dist model
def mh_comp(bd, hps, ss, data):
    """ 
    MH a single component. Return new ss
    """
    
    localss = copy.deepcopy(ss)
    COMP_K = hps['comp_k']
    for comp_i in range(COMP_K):
        pre_score = bd.data_prob(hps, localss, data)
        # mh the mu
        old_mu = ss['mu'][comp_i]
        new_mu = np.random.normal(0, 0.1) + old_mu
        localss['mu'][comp_i] = new_mu
        post_score = bd.data_prob(hps, localss, data)
        
        a = np.exp(post_score - pre_score)
        #print "old mu=", old_mu, "proposed", new_mu, 
        if np.random.rand() > a:
            # reject
            localss['mu'][comp_i] = ss['mu'][comp_i]
            #print "REJECTED a=", a
        else:
            #print "ACCEPTED a=", a 
            pass

    # new pi
    pre_score = bd.data_prob(hps, localss, data)
    old_pi = localss['pi']
    new_pi = np.random.dirichlet(np.ones(len(old_pi)))
    localss['pi'] = new_pi
    post_score = bd.data_prob(hps, localss, data)
    a = np.exp(post_score - pre_score)
    #print "old mu=", old_mu, "proposed", new_mu, 
    if np.random.rand() > a:
        # reject
        localss['pi'] = old_pi
        #print "REJECTED a=", a
    else:
        print "ACCEPTED a=", a, 'pi=', localss['pi']
        pass


        # pre_score = bd.data_prob(hps, localss, data)
        # # mh the mu
        # old_pi = ss['pi'][comp_i]
        # new_pi = np.random.normal(0, 0.1) + old_pi
        # localss['pi'][comp_i] = new_pi
        # post_score = bd.data_prob(hps, localss, data)
        
        # a = np.exp(post_score - pre_score)
        # print "old pi=", old_pi, "proposed", new_pi, 
        # if np.random.rand() > a or new_pi <= 0.001 or new_pi > 0.999:
        #     # reject
        #     localss['pi'][comp_i] = ss['pi'][comp_i]
        #     print "REJECTED a=", a
        # else:
        #     print "ACCEPTED a=", a 

    return localss

def data_prob_mm(data_vect, mu_vect, sigmasq_vect, pi_vect):
    """
    mixture model data probability
    """

    N = len(data_vect)
    K = len(mu_vect)
    tot_score = 0
    for n in range(N):
        score = -1e100
        for k in range(K):
            mu = mu_vect[k]
            sigmasq = sigmasq_vect[k]
            pi = pi_vect[k]
            s = irm.util.log_norm_dens(data_vect[n], mu, sigmasq)
            scores = np.sum(s) + np.log(pi)
            score = np.logaddexp(score, scores)
        tot_score += score
    return tot_score

def data_prob_mm_fast(data_vect, mu_vect, sigmasq_vect, pi_vect):
    """
    mixture model data probability
    """

    N = len(data_vect)
    K = len(mu_vect)
    tot_score = 0

    dp_per_comp_scores = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        mu = mu_vect[k]
        sigmasq = sigmasq_vect[k]
        pi = pi_vect[k]
        dp_per_comp_scores[:, k] = irm.util.log_norm_dens(data_vect, mu, sigmasq)
        dp_per_comp_scores[:, k] += np.log(pi)

    scores = dp_per_comp_scores[:, 0]
    for k in range(1, K):
        scores = np.logaddexp(scores, dp_per_comp_scores[:, k])
    return np.sum(scores)
