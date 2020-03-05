"""
Module containing the agent classes to solve a Bandit problem.

Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
An example can be seen on the Bandit_Agent and Random_Agent classes.
"""
# -*- coding: utf-8 -*-
import numpy as np
from utils import softmax, my_random_choice

class Bandit_Agent(object):
    """
    Abstract Agent to solve a Bandit problem.

    Contains the methods learn() and act() for the base life cycle of an agent.
    The reset() method reinitializes the agent.
    The minimum requirment to instantiate a child class of Bandit_Agent
    is that it implements the act() method (see Random_Agent).
    """
    def __init__(self, k:int, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        pass

    def learn(self, a:int, r:float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        pass

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        raise NotImplementedError("Calling method act() in Abstract class Bandit_Agent")

class Random_Agent(Bandit_Agent):
    """
    This agent doesn't learn, just acts purely randomly.
    Good baseline to compare to other agents.
    """
    def act(self):
        """
        Random action selection.
        Returns
        -------
        a : positive int < k
            A randomly selected action.
        """
        return np.random.randint(self.k)


class EpsGreedy_SampleAverage(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # This class uses Sample Averages to estimate q; others are non stationary.
    def __init__(self, k:int, **kwargs):
        Bandit_Agent.__init__(self,k)
        self.Q = [0 for i in range(self.k)]
        self.counters = [0 for i in range(self.k)]
        self.eps = kwargs["eps"]

    def act(self):
        # First check if we pick at random:
        if self.eps >= np.random.random():
            return np.random.randint(0,self.k)
        else: # Not random so we go greedy
            max_q = max(self.Q)
            max_inds = [i for i, x in enumerate(self.Q) if x == max_q]
            return np.random.choice(max_inds) # if multiple maximums exist we pick one at random

    def learn(self, a:int, r:float):
        # update Q_a and its associated counter
        Q_a = self.Q[a]
        n = self.counters[a]
        total_a = Q_a * n + r
        self.counters[a] += 1
        self.Q[a] = total_a / (n+1)

    def reset(self):
        self.Q = [0 for i in range(self.k)]
        self.counters = [0 for i in range(self.k)]
        

class EpsGreedy(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # Non stationary agent with q estimating and eps-greedy action selection.
    def __init__(self, k:int, **kwargs):
        Bandit_Agent.__init__(self,k)
        self.Q = [0 for i in range(self.k)]
        self.eps = kwargs["eps"]
        self.lr = kwargs["lr"]

    def act(self):
        # First check if we pick at random:
        if self.eps >= np.random.random():
            return np.random.randint(0,self.k)
        else: # Not random so we go greedy
            max_q = max(self.Q)
            max_inds = [i for i, x in enumerate(self.Q) if x == max_q]
            return np.random.choice(max_inds) # if multiple maximums exist we pick one at random

    def learn(self, a:int, r:float):
        # update Q_a and its associated counter
        Q_a = self.Q[a]
        self.Q[a] = Q_a + self.lr*(r - Q_a)

    def reset(self):
        self.Q = [0 for i in range(self.k)]
        self.counters = [0 for i in range(self.k)]

class OptimisticGreedy(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # Same as above but with optimistic starting values.
    def __init__(self, k:int, **kwargs):
        Bandit_Agent.__init__(self,k)
        self.q0 = kwargs["q0"]
        self.lr = kwargs["lr"]
        self.Q = [self.q0 for i in range(self.k)]

    def act(self):
        max_q = max(self.Q)
        max_inds = [i for i, x in enumerate(self.Q) if x == max_q]
        return np.random.choice(max_inds) # if multiple maximums exist we pick one at random

    def learn(self, a:int, r:float):
        # update Q_a and its associated counter
        Q_a = self.Q[a]
        self.Q[a] = Q_a + self.lr*(r - Q_a)

    def reset(self):
        self.Q = [self.q0 for i in range(self.k)]

class UCB(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    def __init__(self, k:int, **kwargs):
        Bandit_Agent.__init__(self,k)
        self.Q = [0 for i in range(self.k)]
        self.counters = [0 for i in range(self.k)]
        self.c = kwargs["c"]
        self.t = 0

    def act(self):
        # This could be implemented more cleanly. Right now we just let divisions by zero happen in 
        # which case we get a np.nan signifying UCB potential is maximized because no tries have happened
        potentials = [self.Q[i] + self.c*np.sqrt(np.log(self.t)/self.counters[i]) for i in range(self.k)]
        for i,x in enumerate(potentials):
            if np.isnan(x):
                potentials[i] = np.inf
        max_q = max(potentials)
        max_inds = [i for i, x in enumerate(potentials) if x == max_q]
        return np.random.choice(max_inds) # if multiple maximums exist we pick one at random

    def learn(self, a:int, r:float):
        # update Q_a and its associated counter
        Q_a = self.Q[a]
        self.counters[a] += 1
        self.t += 1
        self.Q[a] = Q_a + (r - Q_a)/self.counters[a]

    def reset(self):
        self.Q = [0 for i in range(self.k)]
        self.counters = [0 for i in range(self.k)]
        self.t = 0


class Gradient_Bandit:
    # TODO: implement this class following the formalism above.
    # If you want this to run fast, use the my_random_choice function from
    # utils instead of np.random.choice to sample from the softmax
    # You can also find the softmax function in utils.
    def __init__(self, k:int, **kwargs):
        Bandit_Agent.__init__(self,k)
        self.H = [0 for i in range(self.k)]
        #self.Q = [0 for i in range(self.k)]
        #self.counts = [0 for i in range(self.k)]
        self.alpha = kwargs["alpha"]
        self.baseline = 0
        self.count = 0

    def act(self):
        probs = softmax(self.H)
        return my_random_choice(self.k,probs)

    def learn(self, a:int, r:float):
        #Q_a = self.Q[a]
        #n = self.counts[a]
        #total_a = Q_a * n + r
        #self.counts[a] += 1
        #Q_a = total_a / (n+1)
        #self.Q[a] = Q_a
        probA = softmax(self.H)
        self.baseline = (self.baseline * self.count + r) / (self.count + 1)
        self.count += 1
        for i in range(self.k):
            if i == a:
                self.H[i] = self.H[i] + self.alpha * (r - self.baseline) * (1 - probA[i])
            else:
                self.H[i] = self.H[i] - self.alpha * (r - self.baseline) * probA[i]


    def reset(self):
        self.H = [0 for i in range(self.k)]
        self.Q = [0 for i in range(self.k)]
        self.counts = [0 for i in range(self.k)]

class OptimisticGreedy(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # Same as above but with optimistic starting values.
    def __init__(self, k:int, **kwargs):
        Bandit_Agent.__init__(self,k)
        self.q0 = kwargs["q0"]
        self.lr = kwargs["lr"]
        self.Q = [self.q0 for i in range(self.k)]

    def act(self):
        max_q = max(self.Q)
        max_inds = [i for i, x in enumerate(self.Q) if x == max_q]
        return np.random.choice(max_inds) # if multiple maximums exist we pick one at random

    def learn(self, a:int, r:float):
        # update Q_a and its associated counter
        Q_a = self.Q[a]
        self.Q[a] = Q_a + self.lr*(r - Q_a)

    def reset(self):
        self.Q = [self.q0 for i in range(self.k)]
