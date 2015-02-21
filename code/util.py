import numpy as np
import cPickle as pickle
from matplotlib import pylab
from ruffus import * 
from preprocess import * 
import pandas
import sqlite3

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

# plot spatial distribution of each cell type
# plot area vs cell body distance


def create_adj_mat(con, area_thold, cell_data):
    """
    returns (upper-triangular) contact area matrix, cell_ids in order

    """
    
    df = pandas.io.sql.read_frame("select from_id, to_id, area, sum(area) as contact_area, count(area) as contact_count from contacts  where area < %f group by from_id, to_id" % area_thold, 
                                  con)
    
    CELL_N = len(cell_data)
    id_to_pos = {id: pos for pos, id in enumerate(cell_data.index.values)}

    area_mat = np.zeros((CELL_N, CELL_N), dtype=np.float32)

    for c_i, c_row in df.iterrows():
        i1 = id_to_pos.get(c_row['from_id'], -1)
        i2 = id_to_pos.get(c_row['to_id'], -1)
        if i1 >= 0 and i2 >= 0:
            area_mat[i1, i2] = c_row['contact_area']

 
    return area_mat, cell_data.index.values
            



def draw_manual_color_bar(ax, eval_points, color_func, xoffset, yoffset, 
                          xwidth, yscale, 
                          border=True):
    import matplotlib
    from matplotlib.collections import PatchCollection

    patches = []
    colors = []

    for ystart, yend in zip(eval_points[:-1], eval_points[1:]):
        rect = matplotlib.patches.Rectangle((xoffset, ystart*yscale + yoffset), xwidth, yscale*(yend-ystart), 
                                                edgecolor="none")
        patches.append(rect)
        colors.append(color_func((yend-ystart)/2.0 + ystart))
    
    
    
    p = PatchCollection(patches, edgecolor='none', color=colors)
    
    ax.add_collection(p)
    
    if border:
        border_patch = matplotlib.patches.Rectangle((xoffset, yoffset), xwidth, 
                                                    yscale * (np.max(eval_points) - np.min(eval_points)))
        p = PatchCollection([border_patch], edgecolor='k', facecolor='none')
        ax.add_collection(p)
    



