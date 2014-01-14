import pandas
import numpy as np
from ruffus import * 
import sqlite3
import cPickle as pickle
import irm


"""
Create two sets of circos plots based on the features we extract

Features: 
Soma position in layer
layer profile of synapse 
Spatial extent of dendritic arbor [Y/Z]

Fraction of conatcts at that depth ? Number of contacts? 

"""

MAX_CONTACT_AREA = 5.0

BINS = np.linspace(65, 120, 14)

@files("../preprocess/mouseretina/mouseretina.db", "features.pickle")
def create_features(infile, outfile):
    """ 
    input is sqlite table
    output is data frame with all the features

    """

    # soma depth
    # dendritic arbor depth histogram area
    # dendritic arbor depth histogram count

    
    
    conn = sqlite3.connect(infile)


    cells = pandas.io.sql.read_frame("select c.cell_id, c.type_id, s.x as soma_x, s.z as soma_z, s.y as soma_y, t.coarse as type_coarse from cells as c join somapositions as s on c.cell_id = s.cell_id join types as t on c.type_id = t.type_id", 
                                     conn, index_col='cell_id')


    contacts_df = pandas.io.sql.read_frame("select * from contacts where area < %f and area > %f" % (MAX_CONTACT_AREA, 1.0), 
                                           conn, index_col='id')
    
    # contacts sanity check, make sure there is only ONE way of representing cell A contacts cell B
    canon_set = set()
    for from_id, to_id in zip(contacts_df['from_id'], contacts_df['to_id']):
        if (from_id, to_id) in canon_set:
            assert (to_id, from_id) not in canon_set
        canon_set.add((from_id, to_id))

    def f(group):
        #row = group.irow(0)
        gc = group.copy()
        gc['cell_id'] = group['from_id']
        g2 = group.copy()
        g2['cell_id'] = group['to_id']

        #new_df = group
        #return DataFrame({'class': [row['class']] * row['count']})
        return pandas.concat([gc, g2])
    contacts_df_sym = contacts_df.groupby('from_id', group_keys=False).apply(f)


    def feature_extract(group):
        od = {}
        od['contact_x_mean'] = group['x'].mean()
        od['contact_x_mean_area_weight'] = np.average(group['x'], 
                                                      weights=group['area'])
        od['contact_x_std'] = group['x'].std()
        h, e = np.histogram(group['x'], BINS)
        od['contact_x_hist'] = h
        h, e = np.histogram(group['x'], BINS, weights=group['area'])
        od['contact_area_hist'] = h
        od['contact_y_std'] = group['y'].std()
        od['contact_z_std'] = group['z'].std()
        od['contact_spatial_std'] = np.sqrt(group['y'].var() + group['z'].var())

        return pandas.Series(od)

    #results = []
    #for feature_name, feature_func in features.iteritems():
    #    results.append(contacts_df.groupby('from_id').apply(feature_func))

    s = contacts_df_sym.groupby('cell_id').apply(feature_extract)

    a = cells.join(s)

    # now permute arbitarially
    a['cell_id'] = a.index.values
    a = a.reindex(np.random.permutation(a.index))
    print a.head()
    # yes, we have some orphan cells with NO ONE connected to them
    print "There are", np.sum(a.index.values[np.isnan(a['contact_x_mean'])] ), "orphan cells"
    
    pickle.dump({'featuredf' : a}, 
                open(outfile, 'w'))

@files(create_features, "features.png")
def plot_features(infile, outfile):
    data = pickle.load(open(infile, 'r'))
    df = data['featuredf']
    df = df[np.isfinite(df['contact_x_mean'])]
    cell_assignment = df['type_id']

    circos_p = irm.plots.circos.CircosPlot(cell_assignment,
                                           ideogram_radius="0.5r",
                                           ideogram_thickness="10p")

    pos_min = 40
    pos_max = 120
    pos_r_min = 1.00
    pos_r_max = pos_r_min + 0.25
    ten_um_frac = 10.0/(pos_max - pos_min)

    circos_p.add_plot('heatmap', {'r0' : '0.9r', 
                                  'r1' : '1.0r', 
                                  'stroke_thickness' : 0, 
                                  'min' : 0, 
                                  'max' : 72}, 
                      df['type_id'])


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
    # circos_p.add_plot('scatter', {'r0': '1.28r',
    #                               'r1' : '1.50r', 
    #                               'glyph' : 'circle', 
    #                               'glyph_size' : 5, 
    #                               'color' : 'black', 
    #                               'stroke_thickness' : 0}, 
    #                   df['contact_x_mean'], 
    #                   {'backgrounds' : [('background', {'color': 'vvlgrey', 
    #                                                     'y0' : 0, 
    #                                                     'y1' : 100,})], 
    #                    'axes': [('axis', {'color' : 'vgrey', 
    #                                       'thickness' : 1, 
    #                                       'spacing' : '%fr' % 0.1})]})

    for bi, b in enumerate(BINS[:-1]):
        width = 0.03
        start = 1.25 + width*bi
        end = start + width
        r = [row['contact_area_hist'][bi] for (row_i, row) in df.iterrows()]
        print r
        circos_p.add_plot('heatmap', {'r0' : '%fr' % start, 
                                      'r1' : '%fr' % end, 
                                      'stroke_thickness' : 0}, 
                          r)
        

    # circos_p.add_plot('scatter', {'r0': '1.28r',
    #                               'r1' : '1.50r', 
    #                               'glyph' : 'circle', 
    #                               'glyph_size' : 5, 
    #                               'color' : 'red', 
    #                               'stroke_thickness' : 0}, 
    #                   df['contact_x_mean_area_weight'])

    # circos_p.add_plot('scatter', {'r0': '1.53r',
    #                               'r1' : '1.70r', 
    #                               'glyph' : 'circle', 
    #                               'glyph_size' : 5, 
    #                               'color' : 'black', 
    #                               'stroke_thickness' : 0}, 
    #                   df['contact_x_std'], 
    #                   {'backgrounds' : [('background', {'color': 'vvlblue', 
    #                                                     'y0' : 0, 
    #                                                     'y1' : 100,})], 
    #                    'axes': [('axis', {'color' : 'vgrey', 
    #                                       'thickness' : 1, 
    #                                       'spacing' : '%fr' % 0.1})]})

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

    
                                  
    irm.plots.circos.write(circos_p, outfile)

if __name__ == "__main__":
        
    pipeline_run([create_features, plot_features])
    
