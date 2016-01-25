from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy


class QLearning(Agent):
    def learn(self, s, performance_run=False):
        root = self.representation.root
        Qs = self.representation.Qs(root, s)
        # choose an action
        if performance_run:
            aID = self.performance_policy.choose_action(Qs)
        else:
            aID = self.learning_policy.choose_action(Qs)

        (r, ns, terminal) = self.domain.step(s, aID)

        if not performance_run:
            self.logger.info("Learning here!")

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1):
        super(QLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(0.1)

