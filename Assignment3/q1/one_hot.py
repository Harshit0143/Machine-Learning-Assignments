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



def read_lists(filename):
    Lists = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            Lists.append([float(value) for value in line.split()])
    return Lists[0] , Lists[1] , Lists[2]



def make_save_plot(X ,X_label  ,  Ys , Ylabels ,  title, name , save_folder = "plots"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(len(Ys)):
        plt.plot(X, Ys[i] , marker = 'o', markersize = 4  , color = colours[i] ,  label = Ylabels[i])

    plt.xlabel(X_label)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc = 'center right')

    save_path = os.path.join(save_folder, f"{name}.png")
    plt.savefig(save_path)

    plt.close()



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

    def __init__(self, depth):
        self._depth_ = depth
        self._children_ = []
        self._leaf_class_ = None
        self._median_ = None
        self._split_feature_ = None


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
        self._root_ = Decision_Tree_Node(0)

   

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

        children[0] = Decision_Tree_Node(depth_ch)
     
        self.__build_subtree__(children[0], X0[mask0], X1[mask1])

        mask0 = X0[:, split_feature] > median
        mask1 = X1[:, split_feature] > median
        children[1] = Decision_Tree_Node(depth_ch)
   
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
    

    def tree_dfs(self , node : Decision_Tree_Node , L):
        if (node._split_feature_ == None):
            return 
        L.append(node)
        for child in node._children_:
            self.tree_dfs(child , L)

    def compute_remove_acc(self , node : Decision_Tree_Node , X_val , Y_val):
        curr_split_feature = node._split_feature_
        node._split_feature_ = None
        post_prune_accuracy = self.test_accuracy(X_val , Y_val)
        node._split_feature_ = curr_split_feature
        return post_prune_accuracy

    def prune(self , X_val , Y_val):
        curr_acc = self.test_accuracy(X_val , Y_val)
        itr = 0
        print("Started Pruning")
        while (True):
            tree_nodes = []
            post_prunr_accuracies = []
            self.tree_dfs(self._root_ , tree_nodes)
            post_prune_accuracies = np.array([self.compute_remove_acc(to_remove , X_val , Y_val) for to_remove in tree_nodes])
    
            best_node_id = np.argmax(post_prune_accuracies)

            if (post_prune_accuracies[best_node_id] < curr_acc):
                break
            
            tree_nodes[best_node_id]._split_feature_ = None
            curr_acc = post_prune_accuracies[best_node_id]
            itr += 1
            print(f"Iterations done: {itr}")
            print(f'Accuracy: {curr_acc:.3f}%')
        print(f'Number of iterations of pruning: {itr}')
        

def test_sk_model(clf , X_test , Y_test , message):
    accuracy = clf.score(X_test, Y_test) * 100
    print(f'Accuracy on {message}: {accuracy:.3f}%')
    return accuracy
    





def show_scratch_vs_sk_plots():
    Ys = [train_accuracies_scratch_pre , train_accuracies_sk_pre]
    Ylabels = ["scratch" , "sk_learn"]
    make_save_plot(__MAX_DEPTHS__ , 'Max depth of decision tree',  Ys ,  Ylabels , "training set" , "training_set_pre_pruning")

    Ys = [test_accuracies_scratch_pre , test_accuracies_sk_pre]
    Ylabels = ["scratch" , "sk_learn"]
    make_save_plot(__MAX_DEPTHS__ , 'Max depth of decision tree',  Ys ,  Ylabels , "validation set" , "validation _set_pre_pruning")



def run_Problem_b():
    print('\n___Scratch Decision Tree OneHot encoding____________________________________________________________')
    for _max_depth_ in __MAX_DEPTHS__:
        tree = Decision_Tree()
        tree.fit(X_train , Y_train, _max_depth_)
        train_accuracies_hot_scratch_pre.append(tree.test_accuracy(X_train , Y_train , "Training set"))
        val_accuracies_hot_scratch_pre.append(tree.test_accuracy(X_val , Y_val , "Validation set"))
        test_accuracies_hot_scratch_pre.append(tree.test_accuracy(X_test , Y_test , "Testing set"))
        print()
    
    Ys = [train_accuracies_hot_scratch_pre , val_accuracies_hot_scratch_pre , test_accuracies_hot_scratch_pre]
    make_save_plot(__MAX_DEPTHS__ , 'Max depth of decision tree',  Ys ,  Ylabels , "Scratch accuracy OneHot encoder pre-pruning" , "b_")

def run_Probelm_d_pre_pruning():
    print("\n___sk Learn Decision Tree_pre_pruning____________________________________________________________________")
 
    for _max_depth_ in __MAX_DEPTHS__:
        print(f"Max depth of Decision Tree: {_max_depth_}")
        clf = DecisionTreeClassifier(criterion = "entropy", max_depth = _max_depth_, random_state = 42)
        clf.fit(X_train , Y_train)
        train_accuracies_sk_pre.append(test_sk_model(clf , X_train , Y_train , "Training set"))
        val_accuracies_sk_pre.append(test_sk_model(clf , X_val , Y_val , "Validation set"))
        test_accuracies_sk_pre.append(test_sk_model(clf , X_test , Y_test , "Testing set"))
        print()
       
    Ys = [train_accuracies_sk_pre , val_accuracies_sk_pre , test_accuracies_sk_pre]
    make_save_plot(__MAX_DEPTHS__ , 'Max depth of decision tree',  Ys ,  Ylabels , "sk learn accuracy pre-pruning" , "d_pre_pruning")


def run_Probelm_d_post_pruning():
    print("\n___sk Learn Decision Tree_post_pruning____________________________________________________________________")
    for _ccp_alpha_ in ccp_alphas:
        print(f"ccp_alpha value: {_ccp_alpha_}")
        clf = DecisionTreeClassifier(ccp_alpha = _ccp_alpha_ , random_state = 42)
        clf.fit(X_train, Y_train)
        train_accuracies_sk_post.append(test_sk_model(clf ,X_train , Y_train , "Training set"))
        val_accuracies_sk_post.append(test_sk_model(clf ,X_val , Y_val , "Validation set"))
        test_accuracies_sk_post.append(test_sk_model(clf ,X_test , Y_test , "Testing set"))
        print()

    Ys = [train_accuracies_sk_post ,  val_accuracies_sk_post , test_accuracies_sk_post]
    make_save_plot(ccp_alphas, 'ccp_alpha values',  Ys ,  Ylabels , "sk learn accuracy post-pruning" , "d_post_pruning")
    
def run_Problem_e():
    print("\n___sk Learn Random Forest______________________________________________________________________")
   
    model = RandomForestClassifier(oob_score=True)
    print(PARAM_GRID)
   
 
    split_index = [-1]*len(X_train) + [0]*len(X_val)
    X = np.concatenate((X_train, X_val), axis = 0)
    Y = np.concatenate((Y_train, Y_val), axis = 0)
    pds = PredefinedSplit(test_fold = split_index)
    clf = GridSearchCV(estimator = model, cv = pds, param_grid = PARAM_GRID)
    print("Started ehhaustive search")
    clf.fit(X, Y)
    print("Best Hyperparameters:", clf.best_params_)
    best_model = clf.best_estimator_
    print(f'Best OOB Score: {100 * best_model.oob_score_}%')
    test_sk_model(best_model , X_train , Y_train , "training set")
    test_sk_model(best_model , X_val , Y_val , "validation set")
    test_sk_model(best_model , X_test , Y_test , "test set")







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


    LEAST_CNT_INDEX = 75
   

    __MAX_DEPTHS__ = [i for i in range(46)]
    ###(a)###############################################################################################################
    # train_accuracies_scratch_pre , val_accuracies_scratch_pre , test_accuracies_scratch_pre = read_lists("med.txt")


    ###(b)###############################################################################################################
    
    train_accuracies_hot_scratch_pre  = []
    val_accuracies_hot_scratch_pre = []
    test_accuracies_hot_scratch_pre = []
    run_Problem_b()


    ###(d)###############################################################################################################
    ccp_alphas = [0.001, 0.005 ,  0.01 , 0.05 , 0.1, 0.2 , 0.5]
    train_accuracies_sk_pre = []
    val_accuracies_sk_pre = []
    test_accuracies_sk_pre = []
    train_accuracies_sk_post = []
    val_accuracies_sk_post = []
    test_accuracies_sk_post = []
    # run_Probelm_d_pre_pruning()
    # run_Probelm_d_post_pruning()

    # show_scratch_vs_sk_plots()
    ###(e)###############################################################################################################



    PARAM_GRID = {
    'n_estimators': [50 , 100 , 150 , 200 , 250 , 300 , 350],
    'max_features': [0.1 , 0.3 , 0.5 , 0.7 , 0.9 ,  1.0],
    'min_samples_split': [2 , 4 , 6 , 8 , 10]
    }

    # run_Problem_e()
    



"""
____
Started ehhaustive search
USING ONLY 50

Best Hyperparameters: {'max_features': 0.5, 'min_samples_split': 10, 'n_estimators': 50}
Best OOB Score: {100 * best_model.oob_score_}%
Accuracy on training set: 95.924%
Accuracy on validation set: 95.287%
Accuracy on test set: 71.458%
"""