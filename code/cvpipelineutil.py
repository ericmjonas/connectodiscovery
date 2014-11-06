import numpy as np
import copy
import irm
import time 

DEFAULT_CORES = 8
DEFAULT_RELATION = "ParRelation"

def create_cv_pure(data, meta, 
                   cv_i, cv_config_name, cv_config):
    """ 
    Creates a single cross-validated data set 
    """

    shape = data['relations']['R1']['data'].shape
    N =  shape[0] * shape[1]
    if cv_config['type'] == 'nfold':
        np.random.seed(0) # set the seed

        perm = np.random.permutation(N)
        subset_size = N / cv_config['N']
        subset = perm[cv_i * subset_size:(cv_i+1)*subset_size]
        
        observed = np.ones(N, dtype=np.uint8)
        observed[subset] = 0
        data['relations']['R1']['observed'] = np.reshape(observed, shape)

    else:
        raise Exception("Unknown cv type")
    
    

    meta = copy.deepcopy(meta)
    meta['cv'] = {'cv_i' : cv_i,
                  'cv_config_name' : cv_config_name}
    
    return (data, meta)



def create_init_pure(irm_latent, irm_data, OUT_N, 
                init= None, keep_ground_truth=True):
    """ 
    CONVENTION: when we create N inits, the first is actually 
    initialized from the "ground truth" of the intial init (whatever
    that happened to be)

    # FIXME : add ability to init multiple domains
    """
    irm_latents = []

    rng = irm.RNG()

    irm_model = irm.irmio.create_model_from_data(irm_data, rng=rng)
    for c in range(OUT_N):
        np.random.seed(c)

        latent = copy.deepcopy(irm_latent)

        d_N = len(latent['domains']['d1']['assignment'])
        if init['type'] == 'fixed':
            group_num = init['group_num']

            a = np.arange(d_N) % group_num
            a = np.random.permutation(a)

        elif init['type'] == 'crp':
            alpha = init['alpha']
            a = irm.util.crp_draw(d_N, alpha)
            a = np.random.permutation(a) 
        elif init['type'] == 'truth':
            a = latent['domains']['d1']['assignment']
            
        else:
            raise NotImplementedError("Unknown init type")
            
        if (not keep_ground_truth) or (c > 0) : # first one stays the same
            latent['domains']['d1']['assignment'] = a

        # generate new suffstats, recompute suffstats in light of new assignment

        irm.irmio.set_model_latent(irm_model, latent, rng)

        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=2)

        yield irm.irmio.get_latent(irm_model)


def inference_run(data, latent, 
                  kernel_config, 
                  ITERS, seed, VOLUME_NAME, init_type=None, 
                  fixed_k = False, 
                  latent_samp_freq=20, 
                  relation_class = "Relation", 
                  cores = 1):


    if relation_class == "Relation":
        relation_class = irm.Relation
    elif relation_class == "ParRelation":
        relation_class = irm.ParRelation
    else:
        raise NotImplementedError("unknown relation class %s" % relation_class)

    if cores == 1:
        threadpool = None
    else:
        print "Creating threadpool with", cores, "cores"
        threadpool = irm.pyirm.ThreadPool(cores)

    chain_runner = irm.runner.Runner(latent, data, kernel_config, seed, 
                                     fixed_k = fixed_k, 
                                     relation_class = relation_class,
                                     threadpool = threadpool)

    if init_type != None:
        chain_runner.init(init_type)

    scores = []
    times = []
    latents = {}
    def logger(iter, model, res_data):
        print "Iter", iter
        scores.append(model.total_score())
        times.append(time.time())

        if iter % latent_samp_freq == 0:
            latents[iter] = chain_runner.get_state(include_ss=False)
    chain_runner.run_iters(ITERS, logger)
        
    return scores, chain_runner.get_state(), times, latents


def run_exp_pure(data, init, kernel_config_name, seed, kc):
    # put the filenames in the data

    ITERS = kc['ITERS']
    kernel_config = kc['kernels']
    fixed_k = kc.get('fixed_k', False)
    cores = kc.get('cores', DEFAULT_CORES)
    relation_class = kc.get('relation_class', DEFAULT_RELATION)


    res = inference_run(data, init,
                        kernel_config, 
                        ITERS,
                        seed,
                        fixed_k,
                        relation_class=relation_class,
                        cores = cores)
    


    
    return {
            'res' : res, 
            'kernel_config_name' : kernel_config_name}


        
# def save_rdd_elements(rdd, filename_base):
#     """
#     save each element of the rdd in filename_base + .nnn on the 
#     local disk

#     FIXME: This is really just a staging area to get around
#     instantiating the entire rdd in memory

#     """
#     raise Exception("Use common files")
#     conn = boto.connect_s3()
#     bucket = conn.get_bucket(S3_BUCKET)
#     key_name_base = filename_base

#     def create_key_name(index):
#         return S3_PATH + key_name_base + (".%08d" %  index)

#     def save_s3((obj, index)):
#         a = pickle.dumps(obj)
#         k = boto.s3.key.Key(bucket)
#         k.key = create_key_name(index)
#         k.set_contents_from_string(a)
    

#     # materialize, save
#     SIZE_OF_RDD = rdd.count()
#     rdd.zipWithIndex().foreach(save_s3)


#     # now redownload
#     outfiles = []
#     for i in range(SIZE_OF_RDD):
#         key = bucket.new_key(create_key_name(i))
#         contents = key.get_contents_as_string()
#         filename = filename_base + (".%03d" % i)
#         outfiles.append(filename)
        
#         key.get_contents_to_filename(filename)
    
#     pickle.dump(outfiles, 
#                 open(filename_base, 'w'))


# def s3n_delete(url):
#     raise Exception("Use common files")
#     conn = boto.connect_s3()

#     bucket_name = url[6:].split("/")[0]
#     bucket = conn.get_bucket(bucket_name)
    
#     path = url[6 + 1 + len(bucket_name):]
               
#     delete_key_list = []
#     for key in bucket.list(prefix=path):
#         delete_key_list.append(key)
#         if len(delete_key_list) > 100:
#             bucket.delete_keys(delete_key_list)
#             delete_key_list = []

#     if len(delete_key_list) > 0:
#         bucket.delete_keys(delete_key_list)        

# def s3n_url(f):
#     raise Exception("Use common files")
    
#     url =  "s3n://" + os.path.join(S3_BUCKET, S3_PATH, f)
#     print url
#     return url

def experiment_generator(EXPERIMENTS, CV_CONFIGS, INIT_CONFIGS, get_dataset, td):
    for data_name, cv_config_name, init_config_name, kernel_config_name in EXPERIMENTS:
        data_filename = get_dataset(data_name)[0]

        df = "%s-%s-%s-%s" % (data_name, cv_config_name, init_config_name, kernel_config_name)
        
        out_files = [td(df + x) for x in [ ".samples",
                                       ".cvdata", ".inits"]]
        init_config = INIT_CONFIGS[init_config_name]
        cv_config = CV_CONFIGS[cv_config_name]
        
        yield data_filename, out_files, cv_config_name, init_config_name, kernel_config_name, init_config, cv_config

        
