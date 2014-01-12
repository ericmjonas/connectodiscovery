import numpy as np


"""
This is code that presents a mixture model interface
for testing and other stuff


"""
MODEL_DTYPE = {'BetaBernoulli' : np.uint8, 
               'NormalInverseChiSq' : np.float32, 
               'GammaPoisson' : np.uint32}


def create_mm(feature_desc):
    """
    feature_desc: {feature_name : {'data' : numpy data, 
                                   'model' : model'}}

    shared index domain will be 'd1'
    dummy domain will be d_feature_name
    relation will be r_feature_name
    """

    N = len(feature_desc[feature_desc.keys()[0]]['data'])
    latent = {'domains' : {'d1' : {'hps' : {'alpha' : 1.0}, 
                                   'assignment' : np.zeros(N, dtype=np.int32) }}, 
              'relations' : {}}
    data = {'domains' : {'d1': {'N' : N}}, 
            'relations' : {}}
    
    for feature_name, desc in feature_desc.iteritems():
        relation_name = "r_%s" % feature_name
        dummy_domain_name = "d_%s" % feature_name

        # model data sanity check
        if desc['data'].dtype != MODEL_DTYPE[desc['model']] :
            raise Exception("Dtype is incorrect for feature %s" % feature_name)

        data['relations'][relation_name] = {'relation' : ('d1', dummy_domain_name), 
                                            'model' : desc['model'], 
                                            'data' : desc['data']}
        latent['domains'][dummy_domain_name] ={'hps' : {'alpha' : 1.0}, 
                                               'assignment' : [0]}
        latent['relations'][relation_name] = {}
        data['domains'][dummy_domain_name] = {'N' : 1}

        if len(desc['data']) != N:
            raise Exception("Unequal featur length: %s is not of length %d" % (feature_name, N))



    return latent, data
                                              
