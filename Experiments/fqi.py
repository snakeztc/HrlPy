from Domains.Taxi import Taxi
from Trees import Tree1
from Agents.QLearning import QLearning
import numpy as np

taxi_evn = Taxi()
tree1 = Tree1(taxi_evn)
representation = tree1.representation
agent = QLearning(domain=taxi_evn, representation=representation)
max_episode = 10
avg_reward = []

for i in range(0, max_episode):
    s = taxi_evn.s0()
    step_cnt = 0
    cum_reward = 0
    print "Episode " + str(i)
    while True:
        (r, ns, terminal) = agent.execute(s, performance_run=False)
        step_cnt += 1
        cum_reward += r
        if step_cnt >= taxi_evn.episodeCap or terminal:
            break
    avg_reward.append(cum_reward)

print avg_reward
