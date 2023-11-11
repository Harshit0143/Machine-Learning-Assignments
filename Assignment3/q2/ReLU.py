#!/usr/bin/env python3
import numpy as np 
import sys
import pdb
import time

from collections import deque
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
import os
import sys

def _ReLU_(x):
    return max(0 , x)
def _ReLU_div_(y):
    return 1 if y > 0 else 0




def get_data(path_X, path_Y):
    X = np.load(path_X).astype(np.float64)
    Y = np.load(path_Y).astype(np.int64)
    X = 2 * (0.5 - X / 255)
    X = np.hstack((X , np.ones((X.shape[0], 1) , dtype = np.float64)))
    return X , Y

def get_raw_data(path_X, path_Y):
    X = np.load(path_X).astype(np.float64)
    Y = np.load(path_Y).astype(np.int64)
    X = 2 * (0.5 - X / 255)
    return X , Y

def batch_selector(X , Y , M , i):
   return  X[i * M : (i + 1) * M , :] , Y[i * M : (i + 1) * M , :]

### new
class Neural_Network:
    def __init__(self , hidden_layer_arch):
        self.L = len(hidden_layer_arch) + 1
        self.layer_arch = [_ for _ in hidden_layer_arch] 
        self.layer_arch.insert(0 , None) # input layer
        
    
    def fit(self , X_train , Y_train , M , learning_rate):
    
        self.r = Y_train.shape[1]
        self.n = X_train.shape[1]
    

        self.layer_arch[0] = X_train.shape[1]
        self.layer_arch.append(self.r)

        self.theta = [np.zeros((self.layer_arch[i] , self.layer_arch[i - 1]) , dtype = np.float64) for i in range(1 , self.L + 1)]
        self.theta.insert(0 , None)  
    

        self.stochastic_descent_invscaling(X_train , Y_train , M , learning_rate)
     

        
    def propagate_forward(self , X_train): # single example row 
        output  = [None] * (self.L + 1)
        output[0] = X_train  
        for i in range(1 , self.L):
            output[i] = np.matmul(output[i - 1] , self.theta[i].T)    
            output[i]  =  ReLU(output[i])

        
        oL= np.matmul(output[self.L - 1] , self.theta[self.L].T)
        oL = np.exp(oL)
        normalize = np.sum(oL , axis = 1)
        oL = oL / normalize[:, np.newaxis]
        output[self.L] = oL
        return output
  
    def propagate_backward(self , output , Y_train):
        del_ = [None] * (self.L + 1)
        del_[self.L] = Y_train - output[self.L]
        for i in range(self.L - 1 , 0 , -1):
            del_[i] = np.matmul(del_[i + 1] , self.theta[i + 1])    
            del_[i] *= ReLU_div(output[i])
        return del_

    def get_gradient(self , X_train  , Y_train):
        M = Y_train.shape[0]
        output = self.propagate_forward(X_train)
        del_ = self.propagate_backward(output , Y_train)
        J_curr  = np.sum(-np.log(np.sum(output[-1] * Y_train , axis = 1))) / M
       
        return J_curr , np.concatenate([(np.matmul(-del_[flag].T , output[flag - 1]) / M).flatten() for flag in range(1 , self.L + 1)])



    def transformed_matrix(self , arr):
        curr = 0 
        result = [None]
        for i in range(1 , self.L + 1):
            r  , c = self.layer_arch[i] , self.layer_arch[i - 1]
            result.append(arr[curr : curr + r * c].reshape((r , c)))
            curr += r * c
        return result
        

    def perform_iteration(self , X_train , Y_train , M , learning_rate , i):
        
        X , Y  = batch_selector(X_train , Y_train , M , i)
        J_curr , result = self.get_gradient(X , Y)
        result = self.transformed_matrix(result)
        for flag in range(1 , self.L + 1):
            self.theta[flag] -= learning_rate * result[flag]
        return J_curr
        
            
  

    
    def stochastic_descent_invscaling(self , X_train , Y_train , M , learning_rate_0):  

        print('Using Inviscaled learning rate:')
        print('Batch Size in Stochastic Gradient Descent' , M)
        TOLERANCE = 1e-10
        K_PREV = Y_train.shape[0] // M 
        MAX_EPOCH = 500


      
        for i in range(1 , self.L + 1):
            self.theta[i] = np.random.randn(self.layer_arch[i] , self.layer_arch[i - 1]) * FAC[SEED]
    
       
        epoch = 0
        learning_rate = learning_rate_0
        J_prev= np.inf
        while epoch < MAX_EPOCH:

            J_batch = 0
            J_prev= np.inf
            for  _ in range(K_PREV):            
                J_curr =  self.perform_iteration(X_train , Y_train , M , learning_rate , _)
                J_batch += J_curr
                if J_curr == J_prev:
                    with open (f"./dump_re_inv/dump_{SEED}.txt" , 'a') as file:
                        file.write("Retrying\n")
                    self.stochastic_descent_invscaling(X_train , Y_train , M , learning_rate_0)
                    return 
                J_prev = J_curr
            epoch += 1
            J_batch /= K_PREV    

            learning_rate = learning_rate_0 / np.sqrt(epoch)

            with open(f"./dump_re_inv/dump_{SEED}.txt" , 'a') as file:
                file.write(f"Cost: {J_batch}\n")
                file.write(f"Epoch: {epoch}\n")
                file.write(f"Learning Rate: {learning_rate}\n")

            if epoch % 50 == 0:
                with open(f"./acc_re_inv/acc_{SEED}.txt" , 'a') as file:
                    file.write(f"Hidden layer Architecture {self.layer_arch}\n")
                    file.write(f"epoch: {epoch}\n")
                self.test_accuracy(_X_train_ , _Y_train_ , "Training")
                self.test_accuracy(_X_test_ , _Y_test_ , "Testing")
                with open(f"./acc_re_inv/acc_{SEED}.txt" , 'a') as file:
                    file.write("\n\n\n")


            # if epoch > 500 and J_prev > J_batch and J_prev - J_batch < 0.00005:
            #     print("Converged")
            #     break
            # Jprev = J_batch
  
             

    def test_accuracy(self , X_test__ , Y_test__ , message):
        class_probabilities = self.propagate_forward(X_test__)[-1]
        Y_pred = np.argmax(class_probabilities, axis = 1) + 1
        with open(f"./acc_re_inv/acc_{SEED}.txt" , 'a') as file:
            file.write(message + '\n')
            file.write(str(classification_report(Y_pred, Y_test__ , digits = 5)))
            file.write('\n')
        return 100 * np.sum(Y_pred == Y_test__) / len(Y_test__)

  
    def __predict__(self , X):
        class_probabilities = self.propagate_forward(X)[-1]
        predicted_class = np.argmax(class_probabilities)
        class_probabilities = np.zeros(class_probabilities.shape, dtype = np.float64)
        class_probabilities[predicted_class] = 1.0
        return class_probabilities
        
        
        
       


        
       
def train_test_arch(hidden_layer_arch):
    graph = Neural_Network(hidden_layer_arch)
    graph.fit(_X_train_ , Y_train_OneHot , 40 , 0.01)
    graph.test_accuracy(_X_train_ , _Y_train_ , "Training")
    graph.test_accuracy(_X_test_ , _Y_test_ , "Testing")


     

def run_Problem_e():
    # hidden_layer_archs = [[512 , 256]]

    global SEED 
    print("Rectified Linear Activation")
    hidden_layer_archs = [[512, 256, 128, 64] , [512] , [512, 256] , [512, 256, 128]]
    # hidden_layer_archs =  [  [512, 256, 128, 64]]
    for hidden_layer_arch in hidden_layer_archs: 
        output_file = f"./dump_re_inv/dump_{SEED}.txt"
        with open(output_file, "w") as file:
            file.write(f'Hidden layers architecture: {hidden_layer_arch}\n')
        train_test_arch(hidden_layer_arch)
        SEED += 1

def run_Problem_f(): 
    print()
    X_train , Y_train = get_raw_data(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_test , Y_test = get_raw_data(X_TEST_PATH , Y_TEST_PATH)
    print("Shapes")
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    hidden_layer_archs = [(512), (512, 256), (512, 256, 128), (512, 256, 128, 64)]
    with open("ReLU_final_2.txt" , 'a') as file:
        for hidden_layer_arch in hidden_layer_archs:
            file.write(f'Hidden layers architecture: {hidden_layer_arch}\n')
            file.write(f'Running SK Learn :: MLP\n')
            t_start  = time.time()
            clf = MLPClassifier(hidden_layer_sizes= hidden_layer_arch, activation= 'relu',
                        solver = 'sgd', alpha=0, batch_size = 32, learning_rate='invscaling' , max_iter= 1000 , tol = 1e-6)
            clf.fit(_X_train_, _Y_train_)
            file.write(f'Iterations used: {clf.n_iter_}\n')
            
            file.write(f'time taken: {time.time() - t_start} seconds\n')
            file.write("Training data\n")
            Y_pred = clf.predict(_X_train_)
            file.write(f'{classification_report(Y_pred, _Y_train_ , digits = 5)}\n')
            file.write("Testing data")
            Y_pred = clf.predict(_X_test_)
            file.write(f'{classification_report(Y_pred, _Y_test_ , digits = 5)}\n\n')
            print()




 

if __name__ == '__main__':
    X_TRAIN_PATH =  './data/x_train.npy'
    Y_TRAIN_PATH =  './data/y_train.npy'
    X_TEST_PATH =  './data/x_test.npy'
    Y_TEST_PATH =  './data/y_test.npy'
 
    _X_train_ ,_Y_train_ = get_data(X_TRAIN_PATH , Y_TRAIN_PATH)
    _X_test_, _Y_test_ = get_data(X_TEST_PATH , Y_TEST_PATH)
    ReLU = np.vectorize(_ReLU_)
    ReLU_div = np.vectorize(_ReLU_div_)

    FAC = [0.05 , 0.05 , 0.05 , 0.05]
    SEED = 0
    label_encoder = OneHotEncoder(sparse_output = False)
    label_encoder.fit(np.expand_dims(_Y_train_, axis = -1))
    Y_train_OneHot = label_encoder.transform(np.expand_dims(_Y_train_, axis = -1))
    Y_test_OneHot = label_encoder.transform(np.expand_dims(_Y_train_, axis = -1))
    Ylabels = ['training set' , 'test set']
    score_types = ['Accuracy' , 'Precision' , 'Recall' , 'F1_Score']

    colours = ['blue' , 'red' , 'green' , 'magenta', 'black' , 'orange' , 'purple' ,'pink' , 'brown']

    print('\n\n___Problem(e)________________________________________________________________________________')
    # run_Problem_e()
    print('\n\n___Problem(f)________________________________________________________________________________')
    # run_Problem_f()
 
     
     



    