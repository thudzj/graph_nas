import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torchvision
import random
#from visualize import make_dot
from torch.nn.parameter import Parameter
from utils import *
import argparse
import copy
import signal
import sys
# from model import resnet
from controllers import GCN
from util import normalize, sparse_mx_to_torch_sparse_tensor, dense_to_one_hot, output_results

# sys.path.insert(0, '/home/anubhava/ssd.pytorch/')

constrained = False#True

parser = argparse.ArgumentParser(description='Graphical neural architecture search')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                    help='which dataset to test on')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
args = parser.parse_args()


datasetName = args.dataset
useCuda = args.cuda
loadController = False

datasetInputTensor = None
baseline_acc = 0
modelSavePath = None
controllerSavePath = None
if datasetName is 'mnist':
    print('Using mnist')
    import datasets.mnist as dataset
    torch.cuda.set_device(2)
    datasetInputTensor = torch.Tensor(1, 1, 28, 28)
    model = torch.load('./parent_models/mnistvgg13.net')
    #model = torch.load('./parent_models/lenet_mnist.pth')
    baseline_acc = 0.994
    #baseline_acc = 0.983
    modelSavePath = './protos_mnist/'
    controllerSavePath = './controllers_mnist/lstm_lenet.net'
    controllerLoadPath = './controllers_mnist/lstm_lenet.net'
elif datasetName is 'cifar100':
    print('Using cifar100')
    torch.cuda.set_device(1)
    import datasets.cifar100 as dataset
    baseline_acc = 0.67
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    model = torch.load('./parent_models/resnet34_cifar100.net')
    modelSavePath = './protos_cifar100/'
    controllerSavePath = './controllers_cifar100/lstm_resnet34.net'
    controllerLoadPath = './controllers_cifar100/lstm_resnet34.net'
else:
    torch.cuda.set_device(2)
    print('Using cifar')
    import datasets.cifar10 as dataset
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    model = torch.load('./parent_models/vgg11cifar.net')
    baseline_acc = 0.90
    modelSavePath = './protos_cifar/'
    controllerSavePath = './controllers_cifar/lstm_vgg11.net'
    controllerLoadPath = './controllers_cifar/lstm_vggcifar.net'

dataset.args.cuda = useCuda
# parentSize = numParams(model)


# number of different ops
K = 9 # identity  conv_3*3  conv_5*5  max_pooling_3*3  max_pooling_3*3_2  average_pooling_3*3  average_pooling_3*3_2  add  concat

# Parameters for LSTM controller
num_layers = 2
num_hidden = 100
num_input = K
num_output = K + 1
dropout = 0.0
T = 5

controller = GCN(num_input, num_hidden, num_output, num_layers, dropout)
if loadController:
    controller = torch.load(controllerLoadPath)
opti = optim.Adam(controller.parameters(), lr=0.003)

previousModels = {}
# Store statistics for each model
accsPerModel = {}
paramsPerModel = {}
rewardsPerModel = {}
numSavedModels = 0

R_sum = 0
b = 0

'''
Update architecture based on the prediction of last timestep
'''
def update_architecture(adj, f, a):
    add_index = -1
    concat_index = -1
    for i, item in enumate(action):
        if item[-1] != 1:
            if item[-3] == 1: # add op
                if add_index == -1:
                    add_index = adj.shape[1]
                    adj = np.concatenate([adj, dense_to_one_hot(i, adj.shape[0]).transpose()], 1)
                    f = np.concatenate([f, item[:-1]])
                else:
                    adj[i, add_index] = 1
            elif item[-2] == 1: # concat op
                if concat_index == -1:
                    concat_index = adj.shape[1]
                    adj = np.concatenate([adj, dense_to_one_hot(i, adj.shape[0]).transpose()], 1)
                    f = np.concatenate([f, item[:-1]])
                else:
                    adj[i. concat_index] = 1
            else:
                adj = np.concatenate([adj, dense_to_one_hot(i, adj.shape[0]).transpose()], 1)
                f = np.concatenate([f, item[:-1]])
    return adj, f

'''
    Build child model
'''
def build_child_model(adj, f):
    return (adj, f)

def naive_estimate(architecture):
    (adj, f) = architecture
    if np.sum(adj) > 10:
        return 1
    else:
        return -1


def rolloutActions():
    global controller

    actions = []
    adj = sp.coo_matrix((np.array([0]), (np.array([0]), np.array([0]))), shape=(1, 1), dtype=np.float32)
    features = sp.csr_matrix(dense_to_one_hot(np.array([0]), K) , dtype=np.float32)
    for ite in range(T):
        adj_torch = sparse_mx_to_torch_sparse_tensor(normalize(adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])))
        features_torch = torch.FloatTensor(np.array(normalize(features).todense()))
        action = controller(features_torch, adj_torch)
        actions.append(action)
        adj, features = update_architecture(adj, features, action.numpy())
    return actions, build_child_model(adj, features)

def rollout_batch(N, e):
    
    newModels = []
    idxs = []
    Rs = [0]*N
    actionSeqs = []
    for i in range(N):
        
        actions, newModel = rolloutActions()
        actionSeqs.append(actions)
        hashcode = hash(str(newModel)) if newModel else 0
        if hashcode in previousModels and constrained == False:
            Rs[i] = previousModels[hashcode]
        elif newModel is None:
            Rs[i] = -1
        else:
            print(newModel)
            torch.save(newModel, modelSavePath + '%f_%f.net' % (e, i))
            newModels.append(newModel)
            studentModels.append(newModel)
            idxs.append(i)
    #accs = trainTeacherStudentParallel(model, newModels, dataset, epochs=5)
        
    for i in range(len(newModels)):
        #print('Val accuracy: %f' % accs[i])
        #print('Compression: %f' % (1.0 - (float(numParams(newModels[i]))/parentSize)))
        reward = naive_estimate(newModels[i])
        #Reward(accs[i], numParams(newModels[i]), baseline_acc, parentSize, iter=int(e), constrained=constrained, vars=[numParams(newModels[i])], cons=[1700000])
        Rs[idxs[i]] = reward
        hashcode = hash(str(newModel))
        previousModels[hashcode] = reward

    for i in range(len(Rs)):
        print('Reward achieved %f' % Rs[i])
    return (Rs, actionSeqs, newModels)


def rollouts(N, model, e):
    Rs = []
    actionSeqs = []
    models = []
    (Rs, actionSeqs, models) = rollout_batch(N, e)
    return (Rs, actionSeqs, models)


def update_controller(actionSeqs, avgR):
    global b
    print('Reinforcing for epoch %d' % e)
    opti.zero_grad()
    for actions in actionSeqs:
        for action in actions:
            action.reinforce(avgR - b)
        autograd.backward(actions, [None for _ in actions])
    opti.step()

epochs = 100
N = 5
prevRs = [0, 0, 0, 0, 0]
for e in range(epochs):
    # Compute N rollouts
    (Rs, actionSeqs, models) = rollouts(N, model, e)
    # Compute average reward
    avgR = np.mean(Rs)
    print('Average reward: %f' % avgR)
    #b = np.mean(prevRs[-5:])
    prevRs.append(avgR)
    b = R_sum/float(e+1)
    R_sum = R_sum + avgR
    # Update controller
    update_controller(actionSeqs, avgR)

torch.save(controller, controllerSavePath)
resultsFile = open(modelSavePath + 'results.txt', "w")
output_results(resultsFile, accsPerModel, paramsPerModel, rewardsPerModel)