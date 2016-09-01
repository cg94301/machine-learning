import random
import statistics
import numpy as np
import pandas as pd
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from string import join

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 1.0
        self.log = False
        #self.qd = {}

        # States
        self.states = ['greenforward','greenleft','greenright','redforward','redleft','redright']

        # Q matrix, initialized to 0.0
#        self.q = {'stop': pd.Series(.0, index = self.states),
#                  'forward': pd.Series(.0, index = self.states),
#                  'left': pd.Series(.0, index = self.states),
#                  'right': pd.Series(.0, index = self.states)}
#
#
        self.qinit = {'stop': .0,
                  'forward': .0,
                  'left': .0,
                  'right': .0}
        #self.qdf2 = pd.DataFrame(self.qinit, index = ['init'])
        self.qdf2 = pd.DataFrame()

        #self.qdf = pd.DataFrame(self.qinit, index = ['init'])
        self.qdf = pd.DataFrame()

        # Q* for debug purposes hardcoded
        self.qbest = {'stop': pd.Series([1.0,1.0,1.0,1.0,1.0,1.0], index = self.states),
                      'forward': pd.Series([3.0,0.5,0.5,0.0,0.0,0.0], index = self.states),
                      'left': pd.Series([0.5,3.0,0.5,0.0,0.0,0.0], index = self.states),
                      'right': pd.Series([0.5,0.5,3.0,0.5,0.5,3.0], index = self.states)}
        self.qbestdf = pd.DataFrame(self.qbest)
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.env.totalreward = 0.0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        #state2 = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        state2 = (inputs['light'], self.next_waypoint)

        #state = inputs['light'] + self.next_waypoint
        state = ':'.join((inputs['light'], self.next_waypoint))
        #state3 = inputs['light'] + str(inputs['oncoming']) + str(inputs['left']) + str(inputs['right']) + self.next_waypoint
        state3 = ':'.join((inputs['light'], str(inputs['oncoming']), str(inputs['left']), str(inputs['right']), self.next_waypoint))
        self.state = state
        if self.log: print "[DEBUG] state: %s" % state

        if state3 not in self.qdf2.index:
            newq2 = pd.DataFrame(self.qinit, index = [state3])
            self.qdf2 = self.qdf2.append(newq2)

        if state not in self.qdf.index:
            newq = pd.DataFrame(self.qinit, index = [state])
            self.qdf = self.qdf.append(newq)

        #print len(self.qdf2)
        #print self.qdf2

        # TODO: Select action according to your policy

        # select best action from Q matrix
        qaction = self.qdf.loc[state].idxmax()

        # random selection of action
        randaction = random.choice(self.env.valid_actions)

        # Action Policy
        sample = random.random()
        if self.log: print "[DEBUG] epsilon: %s" % self.epsilon
        if sample <= self.epsilon:
            action = randaction
            if self.log: print "[DEBUG] randomaction: %s" % action
        else:
            if qaction == 'stop':
                action = None
            else:
                action = qaction
            if self.log: print "[DEBUG] bestaction: %s" % action

        # Slowly decrease epsilon, leave at 5% floor
        self.epsilon -= .001
        if self.epsilon <= .05:
            self.epsilon = .05

        # Override action w/ Q* for debug only
        # qbestaction =  self.qbestdf.loc[state].idxmax()
        # if qbestaction == 'stop':
        #     qbestaction = None
        # action = qbestaction
        action = randaction

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if action == None:
            nowstate = self.qdf['stop'][state]
        else:
            nowstate = self.qdf[action][state]

        nextstate = nowstate * ( 1 - self.alpha ) + self.alpha * ( reward + self.gamma * 2.0)
        if self.log: print "[DEBUG] nextstate: %s" % nextstate

        # Update Q dict
        #self.qd[(state2, action)] = (1 - self.alpha) * self.qd.get((state2, action), 0) + self.alpha * reward
        #print "[DEBUG] qd:"
        #print len(self.qd)
        #print self.qd


        # Update Q matrix
        if action == None:
            self.qdf['stop'][state] = nextstate
        else:
            self.qdf[action][state] = nextstate

        if self.log: print "[DEBUG] qdf:"
        if self.log: print self.qdf
        print self.qdf
        
        self.env.totalreward += reward
        if self.log: print "[DEBUG] totalreward: %s" % self.env.totalreward
        if self.log: print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
