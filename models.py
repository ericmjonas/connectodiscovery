import numpy as np
import cPickle as pickle
import irm

# this is code that goes from a dataset to a model (data, latent, meta)


def create_conn_dist(infile, dist_vars, model_name= 'LogisticDistance'):
    
    d = pickle.load(open(infile, 'r'))
    conn_mat = d['conn_mat']
    dist_mats = d['dist_mats']
    
    CELL_N = len(conn_mat)
    conn_dist_matrix = np.zeros((CELL_N, CELL_N), 
                           dtype=[('link', np.uint8), 
                                  ('distance', np.float32)])
    conn_dist_matrix['link'] = conn_mat
    
    for dist_var in dist_vars:
        conn_dist_matrix['distance'] += dist_mats[dist_var]**2.0
    conn_dist_matrix['distance'] = np.sqrt(conn_dist_matrix['distance'])

    irm_latent, irm_data = irm.irmio.default_graph_init(conn_dist_matrix, model_name)

    if model_name == "LogisticDistance":
        HPS = {'mu_hp' : 10.0,
               'lambda_hp' : 10.0,
               'p_min' : 0.05, 
               'p_max' : 0.90}
    else:
        raise NotImplementedError()

    irm_latent['relations']['R1']['hps'] = HPS

    return irm_latent, irm_data
