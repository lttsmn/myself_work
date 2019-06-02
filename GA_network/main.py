"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from CBC_Data import get_result
from wbc_data import get_result_wbc
from wdbc_data import get_result_wdbc
from wpbc_data import get_result_wpbc
from Breast_data import get_result_BreastTossue
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import numpy as np
# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)
def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    acc=[]
    roc_auc=[]
    total_accuracy = 0
    for network in networks:
        acc.append(network.accuracy)
        roc_auc.append(network.auc)

    return acc,roc_auc
def get_average_sen_spe(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    sen = []
    spe=[]
    for network in networks:
        sen.append(network.sen)
        spe.append(network.spe)

    return sen,spe
def generate(generations, population, nn_param_choices, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        acc_yuan,auc_yuan=get_average_accuracy(networks)
        average_accuracy = np.mean(acc_yuan)
        std_acc=np.std(acc_yuan)

        average_auc = np.mean(auc_yuan)
        std_auc=np.std(auc_yuan)

        average_sen=np.mean(get_average_sen_spe(networks)[0])
        std_sen=np.std(get_average_sen_spe(networks)[0])

        average_spe=np.mean(get_average_sen_spe(networks)[1])
        std_spe=np.std(get_average_sen_spe(networks)[1])
        # Print out the average accuracy each generation.
        logging.info("accuracy average: %.2f%%" % (average_accuracy * 100))
        logging.info("accuracy std: %.2f%%" % (std_acc * 100))

        logging.info("sen average: %.2f%%" % (average_sen * 100))
        logging.info("sen std: %.2f%%" % (std_sen * 100))

        logging.info("spe average: %.2f%%" % (average_spe * 100))
        logging.info("spe std: %.2f%%" % (std_spe * 100))

        logging.info("auc average: %.2f%%" % (average_auc * 100))
        logging.info("auc std: %.2f%%" % (std_auc * 100))
        logging.info('-'*80)
        networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
        print_networks(networks)
        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    #l=len(networks)
    #network=networks[l-1]
    #acu_curve(network.test,network.predict)
    # Sort our final population.
    #networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    # Print out the top 5 networksd
    #print("the length of networks is:",len(networks))
    #print_networks(networks)
    #return networks[0:6]
    return networks

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()
def main():

    """Evolve a network."""
    #generations = 10  # Number of times to evole the population.
    #population = 20  # Number of networks in each generation.
    generations=1
    population =20
    dataset = 'wbc'
    #dataset = 'CBC'
    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
        # 'nb_neurons': [256,1024],
        #'nb_layers': [2],
       # 'activation': ['relu'],
        #'optimizer': [ 'adam','adamax'],
    }
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    nets=generate(generations, population, nn_param_choices, dataset)
    if dataset == 'CBC':
        fpr,tpr,fpr1, tpr1,fpr2, tpr2,fpr3, tpr3,roc_auc,roc_auc1,roc_auc2,roc_auc3=get_result()
    elif dataset=="wbc":
        fpr,tpr,fpr1, tpr1,fpr2, tpr2,fpr3, tpr3,roc_auc,roc_auc1,roc_auc2,roc_auc3=get_result_wbc()
    elif dataset=="wdbc":
        fpr,tpr,fpr1, tpr1,fpr2, tpr2,fpr3, tpr3,roc_auc,roc_auc1,roc_auc2,roc_auc3=get_result_wdbc()
    elif dataset=="wpbc":
        fpr,tpr,fpr1, tpr1,fpr2, tpr2,fpr3, tpr3,roc_auc,roc_auc1,roc_auc2,roc_auc3=get_result_wpbc()
    elif dataset=="BreastTossue":
        fpr,tpr,fpr1, tpr1,fpr2, tpr2,fpr3, tpr3,roc_auc,roc_auc1,roc_auc2,roc_auc3=get_result_BreastTossue()
    for network in nets:
        fpr0,tpr0,threshold = roc_curve(network.test,network.predict) ###计算真正率和假正率
        roc_auc0 = auc(fpr0,tpr0) ###计算auc的值
        print("auc is:%d " ,roc_auc0)
        plt.figure()
        lw = 2
        plt.plot(fpr0, tpr0, color='blue',
                     lw=lw, label='GA_MLP (area = %0.3f)' % roc_auc0) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr, tpr, color='green',
                     lw=lw, label='MLP (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr1, tpr1, color='red',
                     lw=lw, label='KNN (area = %0.3f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr2, tpr2, color='skyblue',
                     lw=lw, label='SVM (area = %0.3f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr3, tpr3, color='yellow',
                     lw=lw, label='NB (area = %0.3f)' % roc_auc3) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()




if __name__ == '__main__':
    main()
