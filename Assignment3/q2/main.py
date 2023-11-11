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





def get_data(path_X, path_Y):
    X = np.load(path_X).astype(np.float64)
    Y = np.load(path_Y).astype(np.int64)
    X = 2 * (0.5 - X / 255)
    X = np.hstack((X , np.ones((X.shape[0], 1) , dtype = np.float64)))
    return X , Y

def batch_selector(X , Y , M , i):
   return  X[i * M : (i + 1) * M , :] , Y[i * M : (i + 1) * M , :]



class Neural_Network:
    def __init__(self , hidden_layer_arch):
        self.L = len(hidden_layer_arch) + 1
        self.layer_arch = [_ for _ in hidden_layer_arch] 
        self.layer_arch.insert(0 , None) # input layer
        self.flag = None
    
    def fit(self , X_train , Y_train , M , learning_rate, adaptive):
    
        self.r = Y_train.shape[1]
        self.n = X_train.shape[1]
    

        self.layer_arch[0] = X_train.shape[1]
        self.layer_arch.append(self.r)

        self.theta = [np.zeros((self.layer_arch[i] , self.layer_arch[i - 1]) , dtype = np.float64) for i in range(1 , self.L + 1)]
        self.theta.insert(0 , None)  
        # self.batch_gradient_descent(X_train , Y_train, learning_rate)
        # return 

        if adaptive:
            # self.stochastic_descent_adaptive(X_train , Y_train , M , learning_rate)
            self.stochastic_descent_invscaling(X_train , Y_train , M , learning_rate)

        else:
            # self.batch_gradient_descent(X_train , Y_train ,learning_rate)
            self.stochastic_descent(X_train , Y_train , M , learning_rate)
            
        # theta[layer_index][perceptron_index][feature_index]
        # perceptrons are 0 indexed

        
    def propagate_forward(self , X_train): # single example row 
        output  = [None] * (self.L + 1)
        output[0] = X_train  
        for i in range(1 , self.L):
            output[i] = np.matmul(output[i - 1] , self.theta[i].T)    
            output[i]  =  1 / (1 + np.exp(-output[i]))

        
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
            del_[i] *= output[i] * (1 - output[i])
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
        
    def batch_gradient_descent(self , X_train , Y_train , learning_rate_0):
        print('Using constant learning rate:')
        print('Batch Gradient Descent')

        TOLERANCE = 1e-5
     
        for i in range(1 , self.L + 1):
            # self.theta[i] = np.zeros((self.layer_arch[i] , self.layer_arch[i - 1]) , dtype=np.float64)

            # self.theta[i] = np.random.randn(self.layer_arch[i] , self.layer_arch[i - 1]) * 0.05
    
    
        MAX_ITERATIONS = 100000
        print(X_train.shape)
        print(Y_train.shape)
        itr = 0
        J_prev = np.inf
        learning_rate = learning_rate_0
        while  itr < MAX_ITERATIONS: 
            J_curr , result = self.get_gradient(X_train , Y_train)
            if J_curr > J_prev:
                learning_rate /= 10
                print("Stepped Down")
            elif J_prev - J_curr < TOLERANCE:
                break
            result = self.transformed_matrix(result)
            gradient = 0
            for flag in range(1 , self.L + 1):
                self.theta[flag] -= learning_rate * result[flag]
                gradient = max(gradient , np.max(result[flag]))
            itr += 1
            print(f"itr: {itr}")
            print(f"gradient: {gradient}")
            print(f"cost: {J_curr}")
    

        if (itr == MAX_ITERATIONS):
            print("Warning! Max iterations hit.")
        else:
            print("Converged")
            # print(f"Accuracy: {self.test_accuracy(X_train , _Y_train_) :.3f}")

    def perform_iteration(self , X_train , Y_train , M , learning_rate , i):
        
        X , Y  = batch_selector(X_train , Y_train , M , i)
        J_curr , result = self.get_gradient(X , Y)
        result = self.transformed_matrix(result)
        for flag in range(1 , self.L + 1):
            self.theta[flag] -= learning_rate * result[flag]
        return J_curr
        
            

    def stochastic_descent(self , X_train , Y_train , M , learning_rate_0):
  

        print('Using constant learning rate:')
        print('Batch Size in Stochastic Gradient Descent' , M)
        TOLERANCE = 1e-10
        K_PREV = Y_train.shape[0] // M 
        MAX_ITERATIONS = 2500
        MIN_ITERATION = 400
        global MAX_EPOCH 

      
        for i in range(1 , self.L + 1):
            self.theta[i] = np.random.randn(self.layer_arch[i] , self.layer_arch[i - 1]) * (2 / self.layer_arch[i])
    
       
        epoch = 0
        learning_rate = learning_rate_0
        while epoch < MAX_EPOCH:

            # learning_rate = learning_rate_0 / np.sqrt(epoch + 1)
            J_batch = 0
            for  _ in range(K_PREV):            
                J_batch += self.perform_iteration(X_train , Y_train , M , learning_rate , _)
            epoch += 1
            J_batch /= K_PREV    
            with open(f"./dump_inv/dump_{SEED}.txt" , 'a') as file:
                file.write(f"Cost: {J_batch}\n")
                file.write(f"Epoch: {epoch}\n")
        

            if epoch % 500 == 0:
                with open(f"./acc_inv/acc_{SEED}.txt" , 'a') as file:
                    file.write(f"Hidden layer Architecture {self.layer_arch}\n")
                    file.write(f"epoch: {epoch}\n")
                self.test_accuracy(_X_train_ , _Y_train_ , "Training")
                self.test_accuracy(_X_test_ , _Y_test_ , "Testing")
                with open(f"./acc_inv/acc_{SEED}.txt" , 'a') as file:
                    file.write("\n\n\n")

        
  

    
    def stochastic_descent_invscaling(self , X_train , Y_train , M , learning_rate_0):  

        print('Using Inviscaled learning rate:')
        print('Batch Size in Stochastic Gradient Descent' , M)
        TOLERANCE = 1e-10
        K_PREV = Y_train.shape[0] // M 
        MAX_ITERATIONS = 5000
        MIN_ITERATION = 400
        global MAX_EPOCH 

      
        for i in range(1 , self.L + 1):
            self.theta[i] = np.random.randn(self.layer_arch[i] , self.layer_arch[i - 1]) * (2 / self.layer_arch[i])
    
       
        epoch = 0
        learning_rate = learning_rate_0
        J_prev= np.inf
        while epoch < MAX_EPOCH:

            # learning_rate = learning_rate_0 / np.sqrt(epoch + 1)
            J_batch = 0
            for  _ in range(K_PREV):            
                J_batch += self.perform_iteration(X_train , Y_train , M , learning_rate , _)
            epoch += 1
            J_batch /= K_PREV    

            learning_rate = learning_rate_0 / np.sqrt(epoch)

            with open(f"./dump_inv/dump_{SEED}.txt" , 'a') as file:
                file.write(f"Cost: {J_batch}\n")
                file.write(f"Epoch: {epoch}\n")
                file.write(f"Learning Rate: {learning_rate}\n")

            if epoch % 500 == 0:
                with open(f"./acc_inv/acc_{SEED}.txt" , 'a') as file:
                    file.write(f"Hidden layer Architecture {self.layer_arch}\n")
                    file.write(f"epoch: {epoch}\n")
                self.test_accuracy(_X_train_ , _Y_train_ , "Training")
                self.test_accuracy(_X_test_ , _Y_test_ , "Testing")
                with open(f"./acc_inv/acc_{SEED}.txt" , 'a') as file:
                    file.write("\n\n\n")


            if epoch > 500 and J_prev > J_batch and J_prev - J_batch < 0.00005:
                print("Converged")
                break
            J_prev = J_batch
  
        


     

    def test_accuracy(self , X_test__ , Y_test__ , message):
        class_probabilities = self.propagate_forward(X_test__)[-1]
        Y_pred = np.argmax(class_probabilities, axis = 1) + 1
        with open(f"./acc_inv/acc_{SEED}.txt" , 'a') as file:
            file.write(message + '\n')
            file.write(str(classification_report(Y_pred, Y_test__ , digits = 5)))
            file.write('\n')
        return 100 * np.sum(Y_pred == Y_test__) / len(Y_test__)

  

        
        
       
def train_test_arch(hidden_layer_arch , adapt):
    graph = Neural_Network(hidden_layer_arch)
    graph.fit(_X_train_ , Y_train_OneHot , 40 , 0.01 , adaptive = adapt)
    return 
    graph.test_accuracy(_X_train_ , _Y_train_ , "Training")
    graph.test_accuracy(_X_test_ , _Y_test_ , "Testing")

  
       

def run_Problem_a(X_train , Y_train , Y_train_OneHot, X_test , Y_test):
    global SEED , MAX_EPOCH
    MAX_EPOCH = 8000
    hidden_layers = [100 , 50]
    print('Running Neural Network')
    print(f'Hidden Layer architecture {hidden_layers}')
    train_test_arch(hidden_layers , False)
    SEED += 1

    






def run_Problem_b(X_train , Y_train , Y_train_OneHot, X_test , Y_test):
    global SEED , MAX_EPOCH
    hidden_layer_widths = [1 , 5, 10, 50, 100]
    MAX_EPOCH = 5000
    # hidden_layer_widths = [100]

    train_vals , test_vals = [] , []
    for hidden_layer_width in hidden_layer_widths:
        hidden_layer_arch = [hidden_layer_width]
        output_file = f"./dump_inv/dump_{SEED}.txt"
        with open(output_file, "a") as file:
            file.write(f'Hidden layers architecture: {hidden_layer_arch}\n')
        train_test_arch(hidden_layer_arch , False)
        SEED += 1


def run_Problem_c():
    # hidden_layer_archs = [[512 , 256]]

    global SEED , MAX_EPOCH
    MAX_EPOCH = 2500

    hidden_layer_archs = [[512, 256, 128],[512, 256],[512] , [512, 256, 128, 64]]
    train_vals , test_vals = [] , []
    for hidden_layer_arch in hidden_layer_archs: 
        output_file = f"./dump_inv/dump_{SEED}.txt"
        with open(output_file, "w") as file:
            file.write(f'Hidden layers architecture: {hidden_layer_arch}\n')
        train_test_arch(hidden_layer_arch , False)
        SEED += 1

def run_Problem_d():
    # hidden_layer_archs = [[512 , 256]]

    global SEED 
    

    # hidden_layer_archs = [[512, 256, 128],[512, 256],[512] ]
    hidden_layer_archs =  [  [512, 256, 128, 64]]
    train_vals , test_vals = [] , []
    for hidden_layer_arch in hidden_layer_archs: 
        output_file = f"./dump_inv/dump_{SEED}.txt"
        with open(output_file, "w") as file:
            file.write(f'Hidden layers architecture: {hidden_layer_arch}\n')
        train_test_arch(hidden_layer_arch , True)
        SEED += 1

def run_Problem_f(): 
    print()
    hidden_layer_archs = [(512), (512, 256), (512, 256, 128), (512, 256, 128, 64)]
    with open("ReLU_2.txt" , 'a') as file:
        for hidden_layer_arch in hidden_layer_archs:
            file.write(f'Hidden layers architecture: {hidden_layer_arch}\n')
            file.write(f'Running SK Learn :: MLP\n')
            t_start  = time.time()
            clf = MLPClassifier(hidden_layer_sizes= hidden_layer_arch, activation= 'relu',
                        solver = 'sgd', alpha=0, batch_size = 32, learning_rate='invscaling' , max_iter= 1000)
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




# def run_Problem_d():
#     hidden_layer_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
#     train_vals , test_vals = [] , []
#     for hidden_layer_arch in hidden_layer_archs:
#         print(f'Hidden layers architecture: {hidden_layer_arch}')
#         train_test_arch(hidden_layer_arch , True , train_vals , test_vals)

#     show_accuracies(train_vals , test_vals)
#     ids = [_ for _ in range(len(hidden_layer_archs))]
#     build_plots(ids , 'Hidden layer Architecture' ,  train_vals , test_vals , 'Network Architecture' , 'd')




if __name__ == '__main__':
    X_TRAIN_PATH =  './data/x_train.npy'
    Y_TRAIN_PATH =  './data/y_train.npy'
    X_TEST_PATH =  './data/x_test.npy'
    Y_TEST_PATH =  './data/y_test.npy'
    MAX_EPOCH = 2500
    _X_train_,_Y_train_ = get_data(X_TRAIN_PATH , Y_TRAIN_PATH)
    _X_test_, _Y_test_ = get_data(X_TEST_PATH , Y_TEST_PATH)
    
  

  


    label_encoder = OneHotEncoder(sparse_output = False)
    label_encoder.fit(np.expand_dims(_Y_train_, axis = -1))
    Y_train_OneHot = label_encoder.transform(np.expand_dims(_Y_train_, axis = -1))
    Y_test_OneHot = label_encoder.transform(np.expand_dims(_Y_train_, axis = -1))
    Ylabels = ['training set' , 'test set']
    score_types = ['Accuracy' , 'Precision' , 'Recall' , 'F1_Score']

    colours = ['blue' , 'red' , 'green' , 'magenta', 'black' , 'orange' , 'purple' ,'pink' , 'brown']
    categories = label_encoder.categories_
    print("One-Hot Encoding Categories:") # we assume thoruguhout that hsi remians same
    for category in categories:
        print(category)

    SEED = 3
    print('\n\n___Problem(a)________________________________________________________________________________')
    run_Problem_a(_X_train_ , _Y_train_ , Y_train_OneHot, _X_test_ , _Y_test_)
    print('\n\n___Prob lem(b)________________________________________________________________________________')
    # run_Problem_b(_X_train_ , _Y_train_ , Y_train_OneHot, _X_test_ , _Y_test_)
    print('\n\n___Problem(c)________________________________________________________________________________')
    # run_Problem_c()
    print('\n\n___Problem(d)________________________________________________________________________________')
    # run_Problem_d()
    print('\n\n___Problem(f)________________________________________________________________________________')
    # run_Problem_f()
     





    