#!/usr/bin/env python3
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import sys
import time 

# at each node let's store indices of data points that are saved
# we'll need to access the different data dusets
# we'll use numpy to generate hte subset of data
# then make a fucniton to split into indices
# finction for cntinuous and categorial aspects
# need to calculate the information for that



def make_save_plot(X , X_label  ,  Ys , Ylabels ,  title, name , save_folder = 'plots'):
    global colours
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(len(Ys)):
        plt.plot(X, Ys[i] , marker = 'o', markersize = 1 , linewidth = 1  , color = colours[i] ,  label = Ylabels[i])

    plt.xlabel(X_label)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc = 'center right')

    save_path = os.path.join(save_folder, f'{name}.png')
    plt.savefig(save_path)
    plt.close()



def read_lists(filename):
    Lists = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            Lists.append([float(value) for value in line.split()])
    return Lists[0] , Lists[1] , Lists[2]







def get_np_array_hot_endcoder(file_name , message):
    global label_hot_encoder , need_label_encoding , dont_need_label_encoding
    data = pd.read_csv(file_name)
    data = data.drop('number', axis = 1)
    print(f'{message} data size: {len(data)}')
    
   
    if(label_hot_encoder is None):
        label_hot_encoder = OneHotEncoder(sparse_output = False)
        label_hot_encoder.fit(data[need_label_encoding])
    data1 = pd.DataFrame(label_hot_encoder.transform(data[need_label_encoding]), columns = label_hot_encoder.get_feature_names_out())
    
    
    data2 = data.loc[:, dont_need_label_encoding]
    
    final_data = pd.concat([data1, data2], axis = 1)
    final_data = final_data.drop(['result'] , axis = 1)
    Y = data.loc[:,  'result']
    return  np.array(final_data), np.array(Y , dtype = int)





class Decision_Tree_Node:

    def __init__(self, depth , parent = None):
        self._depth_ = depth
        self._children_ = []
        self._leaf_class_ = None
        self._median_ = None
        self._split_feature_ = None
        self._val0_ = 0
        self._val1_ = 0
        self._val_true_ = 0
        self._size_ = 0
        self._parent_ = parent


def give_entropy(X0, X1):
    cnt_Xj_label = len(X0) + len(X1)
    
    if cnt_Xj_label == 0:
        return np.inf # not splittable 
    if len(X0) == 0 or len(X1) == 0:
        return 0
    p_Y0_Xj_label = len(X0) / cnt_Xj_label
    p_Y1_Xj_label = len(X1) / cnt_Xj_label
    H_Y_Xj_0 = -p_Y0_Xj_label * np.log(p_Y0_Xj_label) - p_Y1_Xj_label * np.log(p_Y1_Xj_label)
    return cnt_Xj_label * H_Y_Xj_0 





class Decision_Tree:

    def __init__(self):
        # Tree root should be DTNode
        self._root_ = Decision_Tree_Node(0 , None)
        self.internal = set()
   

    def entropy_split(self, X0, X1,  feature_index):
        
        data_size = len(X0) + len(X1)
        median = 0.5 # categorial
        if feature_index >= LEAST_CNT_INDEX: # continuous
            all_vals = np.concatenate((X0[:, feature_index], X1[:, feature_index]))
            median = np.median(all_vals)
     
        



        H_Y_Xj = 0
        
        X0_ch = X0[X0[:, feature_index] <= median]
        X1_ch = X1[X1[:, feature_index] <= median]
        H_Y_Xj += give_entropy(X0_ch,  X1_ch) / data_size

        X0_ch = X0[X0[:, feature_index] > median]
        X1_ch = X1[X1[:, feature_index] > median]
        H_Y_Xj += give_entropy(X0_ch,  X1_ch) / data_size
        return H_Y_Xj

    def get_best_split(self, X0, X1):
        split_entropies = np.array([self.entropy_split(X0, X1, i) for i in range(self.feature_num)])
        entropy_min_arg = np.argmin(split_entropies)    
        return None if split_entropies[entropy_min_arg] == np.inf else entropy_min_arg

  
    def get_children_split(self, node : Decision_Tree_Node ,  X0, X1,  split_feature,  depth_ch):   
        children = [None , None] 
        median = 0.5 # categorial
        if split_feature >= LEAST_CNT_INDEX: # continuous
            all_vals = np.concatenate((X0[:, split_feature], X1[:, split_feature]))
            median = np.median(all_vals)
         

      


        node._median_ = median

        mask0 = X0[:, split_feature] <= median
        mask1 = X1[:, split_feature] <= median
        children[0] = Decision_Tree_Node(depth_ch , node)
        self.__build_subtree__(children[0], X0[mask0], X1[mask1])
     
        mask0 = X0[:, split_feature] > median
        mask1 = X1[:, split_feature] > median
        children[1] = Decision_Tree_Node(depth_ch , node)

        self.__build_subtree__(children[1], X0[mask0], X1[mask1])

        return children
    
  

    def __build_subtree__(self, node: Decision_Tree_Node, X0, X1):
        pure = len(X0) == 0 or len(X1) == 0
        if (pure or node._depth_ == self.max_depth):
            if len(X0) == 0 and len(X1) == 0:
                node._leaf_class_ = -1 # Never happens. By the Algorithm itself
                print("Unexprected Exception")
                return 

            elif len(X0) >=  len(X1): # if same we predict as 0. Rare
                node._leaf_class_ = 0

            else:
                node._leaf_class_ = 1
            return 
       
        
        node._split_feature_ = self.get_best_split(X0, X1)
        node._leaf_class_ = 0 if len(X0) >= len(X1) else 1
       
        if (node._split_feature_ == None):
            print("Found Not splittable!!!!") # never seen 
            return 
        node._children_ = self.get_children_split(node, X0, X1, node._split_feature_, 1 + node._depth_)
    

    def fit(self, X_train , Y_train , _max_depth_ = np.inf):
        print(f"Max depth of Decision Tree: {_max_depth_}")
      
        self.feature_num = X_train.shape[1]
        self.max_depth = _max_depth_
        mask0 = Y_train == 0
        mask1 = Y_train == 1
        X0 = X_train[mask0, :]
        X1 = X_train[mask1, :]
        self.__build_subtree__(self._root_, X0, X1)
 
    def test_accuracy(self , X_test , Y_test , message = None):
        results = np.apply_along_axis(self.__predict__ , axis = 1, arr = X_test)
        accuracy = accuracy_score(Y_test , results)
        if (message != None ):
            print(f'Accuracy on {message}: {100 * accuracy:.3f}%')
        return 100 * accuracy

    def __predict__(self , X):
        curr = self._root_
        while curr._split_feature_ != None:
            curr = curr._children_[0] if X[curr._split_feature_] <= curr._median_ else curr._children_[1]
        return curr._leaf_class_
    




    def recursive_test(self , node : Decision_Tree_Node , X , Y):
        if node._split_feature_ == None:
            if Y == 0:
                node._val0_ += 1 
            else: 
                node._val1_ += 1
            node._val_true_ +=  (1 if node._leaf_class_ == Y else 0)
        elif  X[node._split_feature_] <= node._median_: 
            self.recursive_test(node._children_[0] , X , Y)
        else:
            self.recursive_test(node._children_[1] , X , Y)

            
    def populate_entire_tree(self, node : Decision_Tree_Node):
        if node._split_feature_ == None:
            return node._val0_ , node._val1_ , node._val_true_
        
        node._val0_ , node._val1_ , node._val_true_ = 0 , 0 , 0
        for i in range(2):
            ch0 , ch1 , ch_true = self.populate_entire_tree(node._children_[i])
            node._val0_ +=  ch0
            node._val1_ +=  ch1
            node._val_true_ +=  ch_true

        return node._val0_ , node._val1_ , node._val_true_


    def tree_dfs(self , node : Decision_Tree_Node):
        if node._split_feature_ != None:    
            self.internal.add(node)
   

            for child in node._children_:
                self.tree_dfs(child)

    def remove_subtree(self , node : Decision_Tree_Node):
        if node._split_feature_ != None:
            self.remove_subtree(node._children_[0])
            self.remove_subtree(node._children_[1])
        self.internal.discard(node) # node either removed from tree or a leaf node
        
    def populate_sizes(self , node: Decision_Tree_Node):
        if node._split_feature_ == None:
            node._size_ = 1
        else:
            size_left = self.populate_sizes(node._children_[0])
            size_right = self.populate_sizes(node._children_[1])
            node._size_ = 1 + size_left + size_right
        return node._size_


            


    def prune(self , X_val , Y_val):
   

        for i in range(len(Y_val)):
            self.recursive_test(self._root_ , X_val[i] , Y_val[i])
        self.populate_entire_tree(self._root_)

        tree_size , train_acc , val_acc , test_acc = [] , [] , [] , []
        self.tree_dfs(self._root_)
        self.populate_sizes(self._root_)
        tree_size.append(self._root_._size_)
        train_acc.append(self.test_accuracy(X_train , Y_train))
        val_acc.append(self.test_accuracy(X_val , Y_val))
        test_acc.append(self.test_accuracy(X_test , Y_test))
       
        print(f'Initial size: {tree_size[-1]}')
        print(f'Initial validation: {self._root_._val_true_ * 100 / len(Y_val) :.3f}')

      
        itr = 0
        while (True):
            tree_nodes = list(self.internal)
            delta_cnt = []
        
            for node in tree_nodes:
                val_true_post_prune = node._val0_ if node._leaf_class_ == 0 else node._val1_
                delta  = val_true_post_prune - node._val_true_
                delta_cnt.append(delta)

            delta_cnt = np.array(delta_cnt)
            best_node_id = np.argmax(delta_cnt)

            if (delta_cnt[best_node_id] < 0):
                break

            best_node = tree_nodes[best_node_id] 
            best_delta = delta_cnt[best_node_id]    
            curr = best_node
            delta_size = 1 - best_node._size_
            while curr != None:
                curr._val_true_ += best_delta
                curr._size_ += delta_size
                curr = curr._parent_
            self.remove_subtree(best_node)
            best_node._split_feature_ = None
            best_node._children_ = []
            itr += 1
        

            tree_size.append(self._root_._size_)
            train_acc.append(self.test_accuracy(X_train , Y_train))
            test_acc.append(self.test_accuracy(X_test , Y_test))
            val_acc.append(self.test_accuracy(X_val , Y_val))







        print(f'Number of iterations of pruning: {itr}')
        print(f'Final validation: {self._root_._val_true_ * 100 / len(Y_val) :.3f}')


        print(f'Final size: {tree_size[-1]}')
        return tree_size , train_acc , val_acc , test_acc
        



def run_Problem_c():
    for max_depth in __MAX_DEPTHS__:
        tree = Decision_Tree()
        tree.fit(X_train , Y_train , max_depth)
        print(f"Max depth {max_depth}")
        print("Pre-pruning accuracies")
        tree.test_accuracy(X_train , Y_train , "training set")
        tree.test_accuracy(X_val , Y_val , "validation set")
        tree.test_accuracy(X_test , Y_test , "test set")
        tree_size , train_acc , val_acc , test_acc = tree.prune(X_val , Y_val)
        print("Post-pruning accuracies:")
        tree.test_accuracy(X_train , Y_train , "training set")
        tree.test_accuracy(X_val , Y_val , "validation set")
        tree.test_accuracy(X_test , Y_test , "test set")
        Ys = [train_acc , val_acc , test_acc]
        Yla = ["Training" , "Validation" , "Testing"]
        make_save_plot(tree_size , "Number of nodes in DTree" , Ys, Yla , f"Pruning tree with max_depth {max_depth} " , f"c_pruning_{max_depth}"  )
        print()


if __name__ == '__main__':
    label_hot_encoder = None
    TRAIN_DATA_PATH = './data/train.csv'
    VAL_DATA_PATH = './data/val.csv'
    TEST_DATA_PATH = './data/test.csv'
    need_label_encoding = ['team', 'opp' , 'host', 'month', 'day_match']
    dont_need_label_encoding = ['toss', 'bat_first', 'format', 'fow', 'year' , 'score', 'rpo', 'result']
    Ylabels = ["trainng set" , "validation set" , "test set"]
    colours = ['blue' , 'red' , 'green' , 'magenta', 'black' , 'orange' , 'purple' ,'pink' , 'brown']

    X_train , Y_train = get_np_array_hot_endcoder(TRAIN_DATA_PATH , "Training")
    X_val , Y_val = get_np_array_hot_endcoder(VAL_DATA_PATH , "Validation")
    X_test ,  Y_test = get_np_array_hot_endcoder(TEST_DATA_PATH , "Testing")
    
    __MAX_DEPTHS__ = [15, 25, 35, 45]

    ###(c)###############################################################################################################
    LEAST_CNT_INDEX = 75
    
    run_Problem_c()
