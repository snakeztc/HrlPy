import numpy as np
from Representation import Representation
import logging


class DagLinear(Representation):

    def phi_sa(self, o, s, u):
        # feature vector
        feature_num = np.sum(self.domain.statespace_limits) * len(self.tree.get(o))
        phi_sa = np.zeros(feature_num)

    def Q(self, o, s, u):
        pass
