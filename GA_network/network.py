"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.sen=0.
        self.spe=0.
        self.auc=0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.test=[]
        self.predict=[]
    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        ac,se,sp,te,pe,auc=train_and_score(self.network, dataset)
        #if self.accuracy == 0.:
        self.accuracy = ac
        #if self.sen==0.:
        self.sen = se
        #if self.spe==0.:
        self.spe = sp
        #if self.auc==0:
        self.auc=auc
        self.test=te
        self.predict=pe
    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
        logging.info("Network sen: %.2f%%" % (self.sen * 100))
        logging.info("Network spe: %.2f%%" % (self.spe* 100))
