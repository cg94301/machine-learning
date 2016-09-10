import random
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
        self.floor = 0.05
        self.log = True

        self.qinit = {'wait': .0, 'forward': .0, 'left': .0, 'right': .0}
        self.qdf = pd.DataFrame()

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

        # Filter state space
        # Can ignore traffic when red light except for potential collision on right turn on red
        # Reason: Crossing red and accident have same penalty
        if inputs['light'] == 'red':
            if inputs['oncoming'] != 'left':
                inputs['oncoming'] = None
            if inputs['left'] != 'forward':
                inputs['left'] = None
            inputs['right'] = None

        if inputs['light'] == 'green':
            # ignore oncoming right turn
            if inputs['oncoming'] == 'right':
                inputs['oncoming'] = None
            # ignore right turn from left
            if inputs['left'] == 'right':
                inputs['left'] = None

        # Select State Space
        #state = ':'.join((inputs['light'], self.next_waypoint))
        state = ':'.join((inputs['light'], str(inputs['oncoming']), str(inputs['left']), str(inputs['right']), self.next_waypoint))

        self.state = state
        if self.log: print "[DEBUG]: %s" % state

        if state not in self.qdf.index:
            newq = pd.DataFrame(self.qinit, index = [state])
            self.qdf = self.qdf.append(newq)

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
            if qaction == 'wait':
                action = None
            else:
                action = qaction
            if self.log: print "[DEBUG] bestaction: %s" % action

        # Slowly decrease epsilon, leave at 5% floor
        if self.epsilon > self.floor:
            self.epsilon -= .00075

        # Override selection. DEBUG only !
        #action = qaction
        #action = randaction

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if action == None:
            nowstate = self.qdf['wait'][state]
        else:
            nowstate = self.qdf[action][state]

        nextstate = nowstate * ( 1 - self.alpha ) + self.alpha * ( reward + self.gamma * 2.0)
        if self.log: print "[DEBUG] nextstate: %s" % nextstate

        # Update Q matrix
        if action == None:
            self.qdf['wait'][state] = nextstate
        else:
            self.qdf[action][state] = nextstate

        if self.log: print "[DEBUG] qdf:"
        if self.log: print self.qdf
        
        self.env.totalreward += reward
        if self.log: print "[DEBUG] totalreward: %s" % self.env.totalreward
        if self.log: print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "Number of states reached %s" % len(a.qdf)
    print a.qdf

if __name__ == '__main__':
    run()
