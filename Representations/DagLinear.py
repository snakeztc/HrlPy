import numpy as np
from Representation import Representation


class DagLinear(Representation):

    state_feature_base = None

    def __init__(self, domain, root, tree, terminals, seed=1):
        super(DagLinear, self).__init__(domain, root, tree, terminals, seed)
        # initialize the model
        self.models = {}
        self.state_features_num = int(np.sum(self.domain.statespace_limits[:, 1]
                                             - self.domain.statespace_limits[:, 0]))
        self.state_feature_base = np.append([0], np.cumsum(self.domain.statespace_limits[:, 1]
                                         - self.domain.statespace_limits[:, 0]))
        for oID in tree.keys():
            feature_num = self.state_features_num * len(self.tree.get(oID))
            self.models[oID] = np.zeros(feature_num)

    def phi_sa(self, o, s, u):
        # feature vector
        phi = self.phi_s(s)
        feature_num = len(phi) * len(self.tree.get(o))
        phi_sa = np.zeros(feature_num)
        u_idx = self.tree.get(o).index(u)
        phi_sa[u_idx*len(phi):(u_idx+1)*len(phi)] = phi
        return phi_sa

    def phi_s(self, s):
        phi = np.zeros(self.state_features_num)
        for idx, s_var in enumerate(s):
            base = self.state_feature_base[idx]
            phi[base + s_var] = 1
        return phi

    def Q(self, o, s, u):
        phi_sa = self.phi_sa(o, s, u)
        q = np.dot(self.models.get(o), phi_sa)
        return q

    def Qs(self, o, s):
        qs = np.zeros(len(self.tree.get(o)))
        for idx, u in enumerate(self.tree.get(o)):
            qs[idx] = self.Q(o, s, u)
        return qs



