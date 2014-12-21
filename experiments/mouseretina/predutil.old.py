import numpy as np
import irm
import models
import glob
import pandas


# NO LONGER USE -- USE THE ONE FROM THE MAIN code/ dir

# def compute_prob_matrix(tgt_latent, tgt_data, model_name='LogisticDistance'):
#     """
#     Compute the probability of a connection at EVERY LOCATION in the matrix

#     Does not depend on the actual observed values of data


#     """
#     ss = tgt_latent['relations']['R1']['ss']
#     ass = tgt_latent['domains']['d1']['assignment']
#     hps = tgt_latent['relations']['R1']['hps']
#     data_conn = tgt_data['relations']['R1']['data']
    
#     N = data_conn.shape[0]
#     pred = np.zeros((N, N))
    
#     for i in range(N):
#         for j in range(N):
#             c1 = ass[i]
#             c2 = ass[j]
#             c = ss[(c1, c2)]
#             if model_name == "LogisticDistance":
#                 dist = data_conn['distance'][i, j]
#                 y = irm.util.logistic(dist, c['mu'], c['lambda']) 
#                 y = y * (hps['p_max'] - hps['p_min']) + hps['p_min']
#             elif model_name == "BetaBernoulliNonConj":
#                 y = c['p']
#             else:
#                 raise NotImplementedError()
#             pred[i, j] = y
#     return pred
