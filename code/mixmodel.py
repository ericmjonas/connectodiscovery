import numpy as np
from scipy.special import erf
import irm

NOT_ASSIGNED = -1

class Feature(object):
    def __init__(self, data, model):
        self.components = {}
        self.assignments = {}
        self.mod = model
        self.data = data
        
        self.hps = self.mod.create_hp()

    def add_entity_to_group(self, group_id, entity_pos):
        ss = self.components[group_id]
        self.mod.add_data(ss, self.data[entity_pos])
        self.assignments[group_id].add(entity_pos)

    def remove_entity_from_group(self, group_id, entity_pos):
        ss = self.components[group_id]
        self.mod.remove_data(ss, self.data[entity_pos])
        self.assignments[group_id].remove(entity_pos)

    def create_group(self, group_id, rng):
        self.components[group_id] = self.mod.create_ss(self.hps, rng)
        self.assignments[group_id] = set()

    def post_pred(self, group_id, entity_pos):
        ss = self.components[group_id]
        return self.mod.pred_prob(self.hps, ss, self.data[entity_pos])
        
    def delete_group(self, group_id):
        del self.components[group_id]
        del self.assignments[group_id]

    def data_prob(self, group_id):
        ss = self.components[group_id]
        ds = [self.data[i] for i in self.assignments[group_id]]
        return self.mod.data_prob(self.hps, ss, ds)


class MixtureModel(object):
    """
    A single handle that we use to glue objects together
    Also computes the CRP
    """

    def __init__(self, ENT_N, features):
        """
        features : {'relationname' : feature_obj}
        where domainname is the name the relation knows this domain as


        """
        self.groups = set()
        self.g_pos = 0
        self.assignments = np.ones(ENT_N, dtype=np.int)
        self.assignments[:] = NOT_ASSIGNED
        self.temp = 1.0

        self.features = features
        self.alpha = 1.0

    def entity_count(self):
        return len(self.assignments)

    def set_hps(self, hps):
        self.alpha = hps['alpha']
    
    def get_hps(self):
        return {'alpha' : self.alpha}

    def get_groups(self):
        return list(self.groups)

    def group_count(self):
        return len(self.groups)

    def set_temp(self, t):
        self.temp = t

    def create_group(self, rng):
        """
        Returns group ID
        """
        
        new_gid = self.g_pos

        [f.create_group(new_gid, rng) for f in self.features.values()]

        self.g_pos += 1
        self.groups.add(new_gid)

        return new_gid
    
    def group_size(self, gid):
        """
        How many entities in this group
        """
        #FIXME slow
        return np.sum(self.assignments == gid)

    def _assigned_entity_count(self):
        return np.sum(self.assignments != NOT_ASSIGNED)

    def get_assignments(self):
        return self.assignments.copy()

    def delete_group(self, group_id):
        #rel_groupid = self.gid_mapping[group_id]
        [f.delete_group(group_id) for f in self.features.values()]
        
        self.groups.remove(group_id)


    def add_entity_to_group(self, group_id, entity_pos):
        assert self.assignments[entity_pos] == NOT_ASSIGNED

        [f.add_entity_to_group(group_id, entity_pos) for f in self.features.values()]

        self.assignments[entity_pos] = group_id

    def remove_entity_from_group(self, entity_pos):
        assert self.assignments[entity_pos] != NOT_ASSIGNED
        group_id = self.assignments[entity_pos]
        
        [f.remove_entity_from_group(group_id, entity_pos) for f in self.features.values()]

        self.assignments[entity_pos] = NOT_ASSIGNED
        return group_id

    def get_prior_score(self):
        count_vect = irm.util.assign_to_counts(self.assignments)
        score = irm.util.crp_score(count_vect, self.alpha)

        return score
        
    def post_pred(self, group_id, entity_pos):
        """
        Combines likelihood and the CRP
        """
        # can't post-pred an assigned row
        assert self.assignments[entity_pos] == NOT_ASSIGNED
        
        scores = [f.post_pred(group_id, entity_pos) for f in self.features.values()]

        gc = self.group_size(group_id)
        assigned_entity_N = self._assigned_entity_count()

        prior_score = irm.util.crp_post_pred(gc, assigned_entity_N+1, self.alpha)
        #print np.sum(scores), prior_score, gc, assigned_entity_N, self.alpha
        return np.sum(scores) + prior_score

    def score(self):
        score = self.get_prior_score()
        for g in self.groups:
            for f in self.features.values():
                score += f.data_prob(g)
        return score

