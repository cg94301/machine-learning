import random
import statistics
import numpy as np
import pandas as pd
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

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
        #self.totalreward = 0.0
        self.states = ['greenforward','greenleft','greenright','redforward','redleft','redright']
        self.q = {'freeze': pd.Series(.0, index = self.states),
                  'forward': pd.Series(.0, index = self.states),
                  'left': pd.Series(.0, index = self.states),
                  'right': pd.Series(.0, index = self.states)}
        self.qdf = pd.DataFrame(self.q)

        self.qbest = {'freeze': pd.Series([1.0,1.0,1.0,1.0,1.0,1.0], index = self.states),
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
        state = inputs['light'] + self.next_waypoint
        self.state = state
        print "[DEBUG] state: %s" % state

        # TODO: Select action according to your policy

        # select best action from Q matrix while training
        qaction = self.qdf.loc[state].idxmax()

        # select best action from static trained Q matrix
        qbestaction =  self.qbestdf.loc[state].idxmax()
        if qbestaction == 'freeze':
            qbestaction = None

        # random selection of action
        randaction = random.choice(self.env.valid_actions)

        # Action Policy
        sample = random.random()
        print "[DEBUG] epsilon: %s" % self.epsilon
        if sample <= self.epsilon:
            action = randaction
            print "[DEBUG] randomaction: %s" % action
        else:
            if qaction == 'freeze':
                action = None
            else:
                action = qaction
            print "[DEBUG] bestaction: %s" % action

        # Override action w/ Q* for debug only
        #action = qbestaction

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if action == None:
            nowstate = self.qdf['freeze'][state]
        else:
            nowstate = self.qdf[action][state]

        nextstate = nowstate * ( 1 - self.alpha ) + self.alpha * ( reward + self.gamma * 2.0)
        print "[DEBUG] nextstate: %s" % nextstate

        # Update Q matrix
        if action == None:
            self.qdf['freeze'][state] = nextstate
        else:
            self.qdf[action][state] = nextstate

        print "[DEBUG] qdf:"
        print self.qdf

        # Slowly decrease epsilon
        # leave at 5% floor
        self.epsilon -= .001
        if self.epsilon <= .05:
            self.epsilon = .05
        
        self.env.totalreward += reward
        print "[DEBUG] totalreward: %s" % self.env.totalreward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
