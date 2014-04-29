import numpy as np
import numpy as np
import cPickle as pickle
import sqlite3
import preprocess
import irm
import models
import glob
from ruffus import * 
import pandas
import process

"""
Ground truth check: 

1. Load ground truth assignments
2. fit BB likelihood
3. Predict each link 
4. measure accuracy


Hand anatomist 
"""



def compute_prob_matrix(tgt_latent, tgt_data):
    ss = tgt_latent['relations']['R1']['ss']
    ass = tgt_latent['domains']['d1']['assignment']
    hps = tgt_latent['relations']['R1']['hps']
    data_conn = tgt_data['relations']['R1']['data']
    
    N = data_conn.shape[0]
    pred = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            c1 = ass[i]
            c2 = ass[j]
            c = ss[(c1, c2)]
            dist = data_conn['distance'][i, j]
            y = irm.util.logistic(dist, c['mu'], c['lambda']) 
            y = y * (hps['p_max'] - hps['p_min']) + hps['p_min']
            pred[i, j] = y
    return pred

THOLDS = [1, 2, 3]
PRED_EVALS= np.logspace(-4, 0, 41) # np.linspace(0, 1.0, 41)

@files("../preprocess/mouseretina/mouseretina.db", ['truth.%d.predlinks.pickle' % t for t in THOLDS])
def create_truth(dbfile, outfiles):
    conn = sqlite3.connect(dbfile)
    for THOLD_i, outfile in zip(THOLDS, outfiles):
        cells, conn_mat, dist_mats = preprocess.create_data(conn, process.THOLDS[THOLD_i])


        irm_latent, irm_data = models.create_conn_dist_lowlevel(conn_mat, dist_mats, 'xyz', model_name="LogisticDistance")

        irm_latent['relations']['R1']['hps'] = {'lambda_hp': 50.0, 'mu_hp': 50.0, 'p_max': 0.9, 'p_min': 0.01}


        irm_latent['domains']['d1']['assignment'] = irm.util.canonicalize_assignment(cells['type_id'])


        irm_model = irm.irmio.create_model_from_data(irm_data)
        rng = irm.RNG()
        irm.irmio.set_model_latent(irm_model, irm_latent, rng)
        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=40)

        learned_latent = irm.irmio.get_latent(irm_model)

        pred = compute_prob_matrix(learned_latent, irm_data)


        pickle.dump({'pred_mat' : pred, 
                    'truth_mat' : irm_data['relations']['R1']['data']['link'],
                    'thold_i' : THOLD_i}, 
                    open(outfile, 'w'))

@files("../preprocess/mouseretina/mouseretina.db", ['truth.%d.bb.predlinks.pickle' % t for t in THOLDS])
def create_truth_bb(dbfile, outfiles):
    conn = sqlite3.connect(dbfile)
    for THOLD_i, outfile in zip(THOLDS, outfiles):
        cells, conn_mat, dist_mats = preprocess.create_data(conn, process.THOLDS[THOLD_i])


        irm_latent, irm_data = models.create_conn_dist_lowlevel(conn_mat, dist_mats, 'xyz', model_name="LogisticDistance")

        irm_latent['relations']['R1']['hps'] = {'lambda_hp': 50.0, 'mu_hp': 50.0, 'p_max': 0.9, 'p_min': 0.01}


        irm_latent['domains']['d1']['assignment'] = irm.util.canonicalize_assignment(cells['type_id'])


        irm_model = irm.irmio.create_model_from_data(irm_data)
        rng = irm.RNG()
        irm.irmio.set_model_latent(irm_model, irm_latent, rng)
        irm.irmio.estimate_suffstats(irm_model, rng, ITERS=40)

        learned_latent = irm.irmio.get_latent(irm_model)

        pred = compute_prob_matrix(learned_latent, irm_data)


        pickle.dump({'pred_mat' : pred, 
                    'truth_mat' : irm_data['relations']['R1']['data']['link'],
                    'thold_i' : THOLD_i}, 
                    open(outfile, 'w'))

@transform(["data/retina.*.ld*.samples", 
            "data/retina.*.srm_*.samples", 
        
        ], 
           suffix(".samples"), ".predlinks.pickle")
def compute_pred_mat(sample_name, outfile):
    s = sample_name.split('-')

    irm_data = pickle.load(open(s[0], 'r'))

    data_conn = irm_data['relations']['R1']['data']

    samples = pickle.load(open(sample_name, 'r'))

    N = len(data_conn)

    truth_mat = irm_data['relations']['R1']['data']['link']
    truth_mat_t_idx = np.argwhere(truth_mat.flatten() > 0).flatten()
    truth_mat_f_idx = np.argwhere(truth_mat.flatten() == 0).flatten()
    chain_n = 0
    outputs = []
    for chain_i, chain in enumerate(samples['chains']):
        if type(chain['scores']) != int:
            irm_latent_samp = chain['state']
            pred = compute_prob_matrix(irm_latent_samp, irm_data)
            pf = pred.flatten()
            for pred_thold  in PRED_EVALS:
                pm = pf[truth_mat_t_idx]
                t_t = np.sum(pm >=  pred_thold)
                t_f = np.sum(pm <= pred_thold)
                pm = pf[truth_mat_f_idx]
                f_t = np.sum(pm >= pred_thold)
                f_f = np.sum(pm <= pred_thold)
                outputs.append({'chain_i' : chain_i, 
                                'score' : chain['scores'][-1], 
                                'pred_thold' : pred_thold, 
                                't_t' : t_t, 
                                't_f' : t_f, 
                                'f_t' : f_t, 
                                'f_f' : f_f, 
                                't_tot' : len(truth_mat_t_idx), 
                                'f_tot' : len(truth_mat_f_idx)
                            })
                


            chain_n += 1
    df = pandas.DataFrame(outputs)
    pickle.dump({'df' : df}, 
                open(outfile, 'w'))

@transform("truth.*.predlinks.pickle", suffix(".pickle"), ".df.pickle")
def compute_truth_pred_mat(infile, outfile):
    d = pickle.load(open(infile, 'r'))


    truth_mat = d['truth_mat']
    truth_mat_t_idx = np.argwhere(truth_mat.flatten() > 0).flatten()
    truth_mat_f_idx = np.argwhere(truth_mat.flatten() == 0).flatten()

    outputs = []
    
    pred = d['pred_mat']
    pf = pred.flatten()
    for pred_thold  in PRED_EVALS:
        pm = pf[truth_mat_t_idx]
        t_t = np.sum(pm >=  pred_thold)
        t_f = np.sum(pm <= pred_thold)
        pm = pf[truth_mat_f_idx]
        f_t = np.sum(pm >= pred_thold)
        f_f = np.sum(pm <= pred_thold)
        outputs.append({ 'truth' : True, 
                        'pred_thold' : pred_thold, 
                        't_t' : t_t, 
                        't_f' : t_f, 
                        'f_t' : f_t, 
                        'f_f' : f_f, 
                        't_tot' : len(truth_mat_t_idx), 
                        'f_tot' : len(truth_mat_f_idx)
                    })

    df = pandas.DataFrame(outputs)
    pickle.dump({'df' : df}, 
                open(outfile, 'w'))


@merge([compute_pred_mat, compute_truth_pred_mat], 
       'predlinks.pickle')
def merge_predlinks(infiles, outfile):
    outd = []
    for f in infiles:
        print "loading", f
        df = pickle.load(open(f, 'r'))['df']
        df['filename'] = f
        outd.append(df)
    df = pandas.concat(outd)
    pickle.dump(df, open(outfile, 'w'))



if __name__ == "__main__":
    pipeline_run([create_truth, compute_pred_mat, 
                  merge_predlinks, 
                  compute_truth_pred_mat, 
                  #summarize_data, 
              ], multiprocess=4)
