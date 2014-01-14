import numpy as np
from ruffus import * 
import cPickle as pickle
import connattribio
import irm
import features

def to_f32(x):
    a = np.array(x).astype(np.float32)
    assert type(a) == np.ndarray
    return a

@files('features.pickle', "assignment.pickle")
def run_inference(infile, outfile):
    data = pickle.load(open(infile, 'r'))
    df = data['featuredf']
    df_vals = df[np.isfinite(df['contact_x_mean'])]

    N = len(df_vals)

    desc = {'soma_x' : {'data' : to_f32(df_vals['soma_x']), 
                          'model' : 'NormalInverseChiSq'}, 
            # 'contact_spatial_std' : {'data' : to_f32(df_vals['contact_spatial_std']),
            #                          'model' : 'NormalInverseChiSq'}, 
    }
    for i, bi in enumerate(features.BINS[:-1]):
        a = np.array([row['contact_area_hist'][i] for row_i, row in df_vals.iterrows()], 
                     dtype=np.float32)
        print a
        desc['contact_x_hist_%d' % i] =  {'data' : a, 
                                          'model' : 'NormalInverseChiSq'}

    latent, data = connattribio.create_mm(desc)
    latent['domains']['d1']['assignment'] = np.arange(N) % 40

    latent, data = irm.data.synth.prior_generate(latent, data)
    
    
    rng = irm.RNG()
    irm_model = irm.irmio.create_model_from_data(data, rng=rng)
    irm.irmio.set_model_latent(irm_model, latent, rng)
    
    kernel_config = irm.runner.default_kernel_anneal()
    kernel_config[0][1]['subkernels'][-1][1]['grids']['NormalInverseChiSq'] = irm.gridgibbshps.default_grid_normal_inverse_chi_sq(mu_scale=10, var_scale=1, GRIDN=10)
    kernel_config[0][1]['subkernels'][-1][1]['grids']['r_soma_x'] = irm.gridgibbshps.default_grid_normal_inverse_chi_sq(mu_scale=10, var_scale=0.1, GRIDN=10)
    

    MAX_ITERS = 200
    for i in range(MAX_ITERS):
        irm.runner.do_inference(irm_model, rng, kernel_config, i)

        new_latent = irm.irmio.get_latent(irm_model)
        a = new_latent['domains']['d1']['assignment']
        
        print irm.util.assign_to_counts(a)
        print "i=", i, "MAX_ITERS=", MAX_ITERS
    
    pickle.dump({'assignment' : a, 
                 'latent' :new_latent,
                 'data' : data}, 
                open(outfile, 'w'))

@files(run_inference, "results.png")
def plot_results(infile, outfile):

    results = pickle.load(open(infile, 'r'))
    a = results['assignment']

    data = pickle.load(open('features.pickle', 'r'))
    df = data['featuredf']
    df = df[np.isfinite(df['contact_x_mean'])]



    circos_p = irm.plots.circos.CircosPlot(a, #df['type_id'], 
                                           ideogram_radius="0.5r",
                                           ideogram_thickness="10p")

    pos_min = 40
    pos_max = 120
    pos_r_min = 1.00
    pos_r_max = pos_r_min + 0.25
    ten_um_frac = 10.0/(pos_max - pos_min)

    circos_p.add_plot('scatter', {'r0' : '%fr' % pos_r_min, 
                                  'r1' : '%fr' % pos_r_max, 
                                  'min' : pos_min, 
                                  'max' : pos_max, 
                                  'glyph' : 'circle', 
                                  'glyph_size' : 5, 
                                  'color' : 'black',
                                  'stroke_thickness' : 0
                                  }, 
                      df['soma_x'], 
                      {'backgrounds' : [('background', {'color': 'vvlgrey', 
                                                        'y0' : pos_min, 
                                                        'y1' : pos_max})],  
                       'axes': [('axis', {'color' : 'vgrey', 
                                          'thickness' : 1, 
                                          'spacing' : '%fr' % ten_um_frac})]})
    circos_p.add_plot('scatter', {'r0': '1.28r',
                                  'r1' : '1.50r', 
                                  'glyph' : 'circle', 
                                  'glyph_size' : 5, 
                                  'color' : 'black', 
                                  'stroke_thickness' : 0}, 
                      df['contact_x_mean_area_weight'], 
                      {'backgrounds' : [('background', {'color': 'vvlgrey', 
                                                        'y0' : 0, 
                                                        'y1' : 100,})], 
                       'axes': [('axis', {'color' : 'vgrey', 
                                          'thickness' : 1, 
                                          'spacing' : '%fr' % 0.1})]})



    circos_p.add_plot('scatter', {'r0': '1.53r',
                                  'r1' : '1.70r', 
                                  'glyph' : 'circle', 
                                  'glyph_size' : 5, 
                                  'color' : 'black', 
                                  'stroke_thickness' : 0}, 
                      df['type_id'], 
                      {'backgrounds' : [('background', {'color': 'vvlblue', 
                                                        'y0' : 0, 
                                                        'y1' : 100,})], 
                       'axes': [('axis', {'color' : 'vgrey', 
                                          'thickness' : 1, 
                                          'spacing' : '%fr' % 0.1})]})

    # circos_p.add_plot('scatter', {'r0': '1.75r',
    #                               'r1' : '1.95r', 
    #                               'glyph' : 'circle', 
    #                               'glyph_size' : 5, 
    #                               'color' : 'black', 
    #                               'stroke_thickness' : 0}, 
    #                   df['contact_spatial_std'], 
    #                   {'backgrounds' : [('background', {'color': 'vvlred', 
    #                                                     'y0' : 0, 
    #                                                     'y1' : 100,})], 
    #                    'axes': [('axis', {'color' : 'vgrey', 
    #                                       'thickness' : 1, 
    #                                       'spacing' : '%fr' % 0.1})]})

    circos_p.add_plot('heatmap', {'r0' : '0.9r', 
                                  'r1' : '1.0r', 
                                  'stroke_thickness' : 0, 
                                  'min' : 0, 
                                  'max' : 72}, 
                      df['type_id'])

    for bi, b in enumerate(features.BINS[:-1]):
        width = 0.02
        start = 1.75 + width*bi
        end = start + width
        r = [row['contact_area_hist'][bi] for (row_i, row) in df.iterrows()]
        circos_p.add_plot('heatmap', {'r0' : '%fr' % start, 
                                      'r1' : '%fr' % end, 
                                      'stroke_thickness' : 0}, 
                          r)

                                  
    irm.plots.circos.write(circos_p, outfile)

    
pipeline_run([run_inference, plot_results])
