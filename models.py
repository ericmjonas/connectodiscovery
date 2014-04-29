import numpy as np
import cPickle as pickle
import irm
import copy


# this is code that goes from a dataset to a model (data, latent, meta)

def create_conn_dist_lowlevel(conn_mat, dist_mats, dist_vars, 
                              model_name= 'LogisticDistance'):

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
    elif model_name == "LogisticDistanceFixedLambda":
        HPS = {'lambda' : 20.0,
               'mu_hp' : 20.0,
               'p_min' : 0.01, 
               'p_scale_alpha_hp' : 1.0, 
               'p_scale_beta_hp' : 1.0}

    else:
        raise NotImplementedError()

    irm_latent['relations']['R1']['hps'] = HPS

    return irm_latent, irm_data

def create_conn_dist(infile, dist_vars, model_name= 'LogisticDistance'):

    d = pickle.load(open(infile, 'r'))
    conn_mat = d['conn_mat']
    dist_mats = d['dist_mats']
    return create_conn_dist_lowlevel(conn_mat, dist_mats, 
                                     dist_vars, model_name)


def create_mixmodeldata(l, x_min, x_max):

    N = len(l)
    # convert into a real list of lists
    contact_x_list = np.zeros(N, dtype=irm.models.MixtureModelDistribution().data_dtype())
    
    for xi, x in enumerate(l):
        if type(x) == float:
            x = []

        y = np.array(x)
        if len(y) > 1024:
            y = np.random.permutation(y)[:1024]

        y = (y - x_min) / (x_max - x_min)
        # normed to [0, 1]
        contact_x_list[xi]['points'][:len(y)] = y
        contact_x_list[xi]['len'] = len(y)
        

    return contact_x_list

def merge_graph_features(g_l, g_d, f_l, f_d, common_domain):
    """
    Take two latents and two data and merge them. 
    
    """

    # do they both have a common domain name for the domain we're doing inference on? 
    if common_domain not in g_d['domains']:
        raise Exception("common domain not in graph data")
    if common_domain not in f_d['domains']:
        raise Exception("common domain not in feature data")

    # do those domains have the same N? 
    if g_d['domains'][common_domain]['N'] != f_d['domains'][common_domain]['N']:
        raise Exception("domains differ on the number of objects")




    # make sure none of the other domains have overlapping names
    g_domain_names = set(g_d['domains'].keys())
    f_domain_names = set(f_d['domains'].keys())

    if g_domain_names.intersection(f_domain_names) != set([common_domain]):
        raise Exception("Overlapping domain names")


    # make sure none of the relations have overlapping names

    g_relation_names = set(g_d['relations'].keys())
    f_relation_names = set(f_d['relations'].keys())

    if g_relation_names.intersection(f_relation_names) != set([]):
        raise Exception("Overlapping relation names")

    # merge the datas :
    out_d = copy.deepcopy(g_d)
    out_d['relations'].update(copy.deepcopy(f_d['relations']))
    out_d['domains'].update(copy.deepcopy(f_d['domains']))

    # merge the latents :
    out_l = copy.deepcopy(g_l)
    out_l['relations'].update(copy.deepcopy(f_l['relations']))
    out_l['domains'].update(copy.deepcopy(f_l['domains']))

    return out_l, out_d

    
