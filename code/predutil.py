import numpy as np
import irm
import models
import glob
import pandas

def compute_prob_matrix(tgt_latent, tgt_data, model_name='LogisticDistance', 
                        relation_name = 'R1', domain_name = "d1"):
    """
    Compute the probability of a connection at EVERY LOCATION in the matrix

    Does not depend on the actual observed values of data


    """
    ss = tgt_latent['relations'][relation_name]['ss']
    ass = tgt_latent['domains'][domain_name]['assignment']
    hps = tgt_latent['relations'][relation_name]['hps']
    data_conn = tgt_data['relations'][relation_name]['data']
    
    N = data_conn.shape[0]
    pred = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            c1 = ass[i]
            c2 = ass[j]
            c = ss[(c1, c2)]
            if model_name == "LogisticDistance":
                dist = data_conn['distance'][i, j]
                y = irm.util.logistic(dist, c['mu'], c['lambda']) 
                y = y * (hps['p_max'] - hps['p_min']) + hps['p_min']
            elif model_name == "BetaBernoulliNonConj":
                y = c['p']
            elif model_name == "LogisticDistancePoisson":
                dist = data_conn['distance'][i, j]
                conj_model = irm.models.LogisticDistancePoisson()
                rate = conj_model.param_eval(dist, c, hps)

                # We then use the rate and poisson cdf to compute P(k >0)
                y = 1.0 - np.exp(-rate)
                
                
            else:
                raise NotImplementedError()
            pred[i, j] = y
    return pred
