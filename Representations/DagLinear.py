import numpy as np
from Representation import Representation
import numpy.matlib



class DagLinear(Representation):

    state_feature_base = None

    def __init__(self, domain, root, tree, terminals, seed=1):
        super(DagLinear, self).__init__(domain, root, tree, terminals, seed)
        # initialize the model
        self.models = {}
        self.state_features_num = int(np.sum(self.domain.statespace_limits[:, 1]
                                             - self.domain.statespace_limits[:, 0]))
        self.state_feature_base = np.append([0], np.cumsum(self.domain.statespace_limits[:-1, 1]
                                         - self.domain.statespace_limits[:-1, 0]))
        for oID in tree.keys():
            feature_num = self.state_features_num * len(self.tree.get(oID))
            self.models[oID] = np.zeros(feature_num)

    def phi_sa(self, o, s, u):
        """
        Get the feature vector for a subtask o at state s with action u
        :param o: the name of subtask
        :param s: the raw state vector
        :param u: the index!! of action in a np array
        :return: the feature vector
        """
        # feature vector
        phi = self.phi_s(s)
        feature_num = phi.shape[1] * len(self.tree.get(o))
        phi_sa = np.zeros((s.shape[0], feature_num))
        for idx in range(0, s.shape[0]):
            u_idx = u[idx]
            phi_sa[idx, u_idx*self.state_features_num:(u_idx+1)*self.state_features_num] = phi[idx]

        return phi_sa

    def phi_s(self, s):
        phi = np.zeros((s.shape[0], self.state_features_num))
        base = np.matlib.repmat(self.state_feature_base, s.shape[0], 1)
        sparse_s = np.matrix(base + s, dtype=np.int8)

        # convert the sparse_s to one-hot phi
        for idx in range(0, s.shape[0]):
            phi[idx, sparse_s[idx, :]] = 1
        return phi

    def Q(self, o, s, u):
        phi_sa = self.phi_sa(o, s, u)
        q = np.dot(phi_sa, self.models.get(o))
        return q

    def Qs(self, o, s):
        qs = np.zeros((s.shape[0], len(self.tree.get(o))))
        for idx, u in enumerate(self.tree.get(o)):
            temp_uIDs = np.ones(s.shape[0]) * idx
            qs[:, idx] = self.Q(o, s, temp_uIDs)
        return qs



