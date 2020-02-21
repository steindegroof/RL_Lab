"""
Module containing the k-armed bandit problem
Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
We expect all classes to follow the Bandit abstract object formalism.
"""
# -*- coding: utf-8 -*-
import numpy as np

class Bandit(object):
    """
    Abstract concept of a Bandit, i.e. Slot Machine, the Agent can pull.

    A Bandit is a distribution over reals.
    The pull() method samples from the distribution to give out a reward.
    """
    def __init__(self, **kwargs):
        """
        Empty for our simple one-armed bandits, without hyperparameters.
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        pass

    def reset(self):
        """
        Reinitializes the distribution.
        """
        pass

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError("Calling method pull() in Abstract class Bandit")


class Gaussian_Bandit(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the Gaussian_Bandit's distribution is a fixed Gaussian.
    def __init__(self, **kwargs):
        Bandit.__init__(self)
        self.mu = kwargs["mu"]
        self.sigma = kwargs["sigma"]

    def reset(self):
        self.mu = np.random.normal(0,1)

    def setMu(self, mu):
        self.mu = mu

    def setSigma(self, sigma):
        self.sigma = sigma

    def pull(self):
        return np.random.normal(self.mu,self.sigma)

class Gaussian_Bandit_NonStat(Gaussian_Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the distribution mean changes each step over time,
    # with increments following N(m=0,std=0.01)
    def __init__(self, **kwargs):
        Gaussian_Bandit.__init__(self,**kwargs)
        self.firstMu = self.mu

    def reset(self):
        self.mu = self.firstMu

    def pull(self):
        self.mu += np.random.normal(0,0.01)
        return Gaussian_Bandit.pull(self)
   

class KBandit:
    # TODO: implement this class following the formalism above.
    # Reminder: The k-armed Bandit is a set of k Bandits.
    # In this case we mean for it to be a set of Gaussian_Bandits.
    def __init__(self, **kwargs):
        self.mu = kwargs["mu"]
        self.sigma = kwargs["sigma"]
        self.bandits = []
        for i in kwargs["k"]:
            self.bandits.append(Gaussian_Bandit(self.mu,self.sigma))

    def __iter__(self):
        return iter(self.bandits)

    def()
   


class KBandit_NonStat:
    # TODO: implement this class following the formalism above.
    # Reminder: Same as KBandit, with non stationary Bandits.
    pass
