import pandas
import numpy as np
from ruffus import * 
import sqlite3
import cPickle as pickle
import sklearn.metrics

MAX_CONTACT_AREA = 5.0

# Take in the connectivity matrix and 
# 1. randomize the ordering of the cells 
# select the N cells we're going to use and take that data frame
# return 3 distance matrices
# return connectivity matrix with real areas to use as subsequent threshold


# if we are going to say "above size X is a synapse" then we need to be consistent
# with it.

# create the python structures we want / need

def create_adj_mat(con, 
                   area_contact_min, 
                   cell_data):
    """

    """
    
    df = pandas.io.sql.read_frame("select from_id, to_id, area, sum(area) as contact_area, count(area) as contact_count from contacts  where area < %f  and area > %f group by from_id, to_id" % (MAX_CONTACT_AREA, area_contact_min), con)
    
    CELL_N = len(cell_data)
    id_to_pos = {id: pos for pos, id in enumerate(cell_data.index.values)}

    area_mat = np.zeros((CELL_N, CELL_N), dtype=np.float32)

    for c_i, c_row in df.iterrows():
        i1 = id_to_pos.get(c_row['from_id'], -1)
        i2 = id_to_pos.get(c_row['to_id'], -1)
        if i1 >= 0 and i2 >= 0:
            area_mat[i1, i2] = c_row['contact_area']

 
    return area_mat
            
def create_contact_x_lists(con, area_thold_min):
    """
    for each cell collapse the list of synapse hists
    """

    contacts_df = pandas.io.sql.read_frame("select * from contacts where area < %f and area > %f" % (MAX_CONTACT_AREA, area_thold_min), 
                                           con, index_col='id')
    
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
        od['contact_x_list'] = group['x'].tolist()
        return pandas.Series(od)

    s = contacts_df_sym.groupby('cell_id').apply(feature_extract)

    return s



def create_data(con, area_thold_min):
    """
    return a data frame with all the cell_type Ids, as well as the type information
    joined in, the hists as list, the connectivity matrix, and then three
    distance matrices, one for x, y, and z
    """

    cells = pandas.io.sql.read_frame("select c.cell_id, s.x, s.z, s.y, c.type_id, t.coarse from cells as c join somapositions as s on c.cell_id = s.cell_id join types as t on c.type_id = t.type_id where s.x is not null", 
                                     con, index_col='cell_id')
    cells = cells.reindex(np.random.permutation(cells.index))


    contact_x_lists = create_contact_x_lists(con, area_thold_min)

    cells = cells.join( contact_x_lists)
    adj_mat = create_adj_mat(con, area_thold_min, cells) > 0 

    # should be in the right "order" 
    
    conn_mat = ((adj_mat + adj_mat.T) > 0).astype(np.uint8)
    
    # distance matrices
    dist_mats = {}

    for dim in ['x', 'y', 'z']:
        x = np.array(cells[dim])
        x.shape = (len(x), 1)
        dist_mats[dim] = sklearn.metrics.pairwise.pairwise_distances(x)
    
    return cells, conn_mat, dist_mats

    
if __name__ == "__main__":
    db = "../preprocess/mouseretina/mouseretina.db"
    
    conn = sqlite3.connect(db)
    create_data(conn, 0.5)
