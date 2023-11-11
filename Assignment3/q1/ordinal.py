#!/usr/bin/env python3
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
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


def make_save_plot(X ,X_label  ,  Ys , Ylabels ,  title, name , save_folder = "plots"):
    global colours
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








def get_np_array(file_name , message):
    global label_encoder, need_label_encoding

    data = pd.read_csv(file_name)
    data = data.drop('number', axis=1)
    print(f'{message} data size: {len(data)}')

    data1 = data[need_label_encoding]

    if (label_encoder is None):
        label_encoder = OrdinalEncoder()
        label_encoder.fit(data1)

    data1 = pd.DataFrame(label_encoder.transform(
        data1), columns=need_label_encoding).astype(int)

    data_cnt = data.loc[:, ['year', 'fow', 'score', 'rpo']]

    need_not_encode_cat = data.loc[:, ['toss', 'bat_first', 'format']]
    data_cat = pd.concat([data1, need_not_encode_cat], axis=1)
   

    cat_label_cnt = [data[col].nunique() for col in need_label_encoding] + [2, 2, 2]
    y = data.loc[:, 'result']
    return cat_label_cnt, np.array(data_cat), np.array(data_cnt), y.to_numpy()


class Decision_Tree_Node:

    def __init__(self, depth):
        self.depth = depth
        self.children = []
        self.leaf_class = None
        self.median = None
        self.split_feature = None


def give_entropy(data_size, X0, X1):
    cnt_Xj_label = len(X0) + len(X1)
    if (cnt_Xj_label == 0): # can't split by this parameter. np.inf addts to anything to give np.inf
        return np.inf
    if (len(X0) == 0 or len(X1) == 0):
        return 0
    p_Y0_Xj_label = len(X0) / cnt_Xj_label
    p_Y1_Xj_label = len(X1) / cnt_Xj_label
    H_Y_Xj_0 = -p_Y0_Xj_label * np.log(p_Y0_Xj_label) - p_Y1_Xj_label * np.log(p_Y1_Xj_label)
    return cnt_Xj_label * H_Y_Xj_0 / data_size





class Decision_Tree:

    def __init__(self):
        # Tree root should be DTNode
        self.root = Decision_Tree_Node(0)

    def entropy_categorial_split(self, X_cat0, X_cat1,  feature_index):
        child_cnt = self.cat_label_cnt[feature_index]
        data_size = len(X_cat0) + len(X_cat1)
        H_Y_Xj = 0

        for label in range(child_cnt):
            X_cat0_ch = X_cat0[X_cat0[:, feature_index] == label]
            X_cat1_ch = X_cat1[X_cat1[:, feature_index] == label]
            H_Y_Xj += give_entropy(data_size, X_cat0_ch,  X_cat1_ch)

        return H_Y_Xj

    def entropy_continuous_split(self, X_cnt0, X_cnt1,  feature_index):
        data_size = len(X_cnt0) + len(X_cnt1)
        all_vals = np.concatenate(
            (X_cnt0[:, feature_index], X_cnt1[:, feature_index]))
        median = np.median(all_vals)
        H_Y_Xj = 0

        X_cnt0_ch = X_cnt0[X_cnt0[:, feature_index] <= median]
        X_cnt1_ch = X_cnt1[X_cnt1[:, feature_index] <= median]
        H_Y_Xj += give_entropy(data_size, X_cnt0_ch,  X_cnt1_ch)

        X_cnt0_ch = X_cnt0[X_cnt0[:, feature_index] > median]
        X_cnt1_ch = X_cnt1[X_cnt1[:, feature_index] > median]
        H_Y_Xj += give_entropy(data_size, X_cnt0_ch,  X_cnt1_ch)

        return H_Y_Xj

    def get_best_split(self, X_cat0, X_cat1, X_cnt0, X_cnt1):
        
        split_cat_entropies = np.array([self.entropy_categorial_split(
            X_cat0, X_cat1, i) for i in range(self.cat_feature_num)])
        # print("Categorial:", split_cat_entropies)
        split_cnt_entropies = np.array([self.entropy_continuous_split(
            X_cnt0, X_cnt1, i) for i in range(self.cnt_feature_num)])
        # print("Continuous:", split_cnt_entropies)

        cat_min_arg = np.argmin(split_cat_entropies)
        cnt_min_arg = np.argmin(split_cnt_entropies)
      
        if np.min(split_cat_entropies) == np.inf and np.min(split_cnt_entropies) == np.inf:
            # this was never encounterd. There is always atleast 1 splittable feature
            return None
        elif split_cat_entropies[cat_min_arg] <= split_cnt_entropies[cnt_min_arg]:
            return (0, cat_min_arg)
        return (1, cnt_min_arg)

    def get_children_split(self, node ,  X_cat0, X_cat1, X_cnt0, X_cnt1,  split_feature,  depth_ch):
        feature_index = split_feature[1]
        child_cnt = self.cat_label_cnt[feature_index] if split_feature[0] == 0 else 2
        children = [None] * child_cnt
        if (split_feature[0] == 0):
            for label in range(child_cnt):
                mask0 = X_cat0[:, feature_index] == label
                mask1 = X_cat1[:, feature_index] == label
                X_cat0_ch = X_cat0[mask0]
                X_cat1_ch = X_cat1[mask1]
                X_cnt0_ch = X_cnt0[mask0]
                X_cnt1_ch = X_cnt1[mask1]
                children[label] = Decision_Tree_Node(depth_ch)
                self.__build_subtree__(
                    children[label], X_cat0_ch, X_cat1_ch, X_cnt0_ch, X_cnt1_ch)
            return children
        all_vals = np.concatenate(
            (X_cnt0[:, feature_index], X_cnt1[:, feature_index]))
        median = np.median(all_vals)
        node.median = median
        mask0 = X_cnt0[:, feature_index] <= median
        mask1 = X_cnt1[:, feature_index] <= median

        children[0] = Decision_Tree_Node(depth_ch)
     
        self.__build_subtree__(
            children[0], X_cat0[mask0], X_cat1[mask1], X_cnt0[mask0], X_cnt1[mask1])

        mask0 = X_cnt0[:, feature_index] > median
        mask1 = X_cnt1[:, feature_index] > median
        children[1] = Decision_Tree_Node(depth_ch)
   
        self.__build_subtree__(children[1], X_cat0[mask0], X_cat1[mask1], X_cnt0[mask0], X_cnt1[mask1])

        return children

    def __build_subtree__(self, node: Decision_Tree_Node, X_cat0, X_cat1, X_cnt0, X_cnt1):

        pure = len(X_cat0) == 0 or len(X_cat1) == 0
        if (pure or node.depth == self.max_depth):
            if len(X_cat0) == 0 and len(X_cat1) == 0:
                node.leaf_class = -1 # Never happens. By the Algorithm itself
                print("Unexprected Exception")
                return 

            elif len(X_cat0) >=  len(X_cat1):
                node.leaf_class = 0

            else:
                node.leaf_class = 1
            return 
        
    
        node.split_feature = self.get_best_split(X_cat0, X_cat1, X_cnt0, X_cnt1)
       
        if (node.split_feature == None):
            print("Not splittable") # Never seen 
            print(node.split_feature)
            return 

        node.leaf_class = 0 if len(X_cat0) >= len(X_cat1) else 1
        node.children = self.get_children_split(node,
            X_cat0, X_cat1, X_cnt0, X_cnt1, node.split_feature, 1 + node.depth)
        

    def fit(self, _cat_label_cnt_, _X_cat_,  _X_cnt_, _Y_ , max_depth = 10):
        print(f"Max depth of Decision Tree: {max_depth}")
        self.cat_feature_num = _X_cat_.shape[1]
        self.cnt_feature_num = _X_cnt_.shape[1]
        self.cat_label_cnt = _cat_label_cnt_
        self.max_depth = max_depth
        mask0 = _Y_ == 0
        mask1 = _Y_ == 1
        X_cat0 = _X_cat_[mask0, :]
        X_cat1 = _X_cat_[mask1, :]
        X_cnt0 = _X_cnt_[mask0, :]
        X_cnt1 = _X_cnt_[mask1, :]
        self.__build_subtree__(self.root, X_cat0, X_cat1, X_cnt0, X_cnt1)

    def test_accuracy(self , X_test_cat , X_test_cnt , Y_test , message):
        X_test = np.concatenate((X_test_cat, X_test_cnt), axis = 1)
        results = np.apply_along_axis(self.__predict__ , axis = 1, arr = X_test)
        accuracy = accuracy_score(Y_test , results)
        print(f'Accuracy on {message}: {100 * accuracy:.3f}%')
        return 100 * accuracy

    def __predict__(self , X):
        X_cat = X[0: self.cat_feature_num].astype(int)
        X_cnt = X[self.cat_feature_num:]
        curr = self.root
        while curr.split_feature != None:
            (typ , idx) = curr.split_feature
            if typ == 0:
                curr = curr.children[X_cat[idx]]
            elif X_cnt[idx] <= curr.median:
                curr = curr.children[0]
            else:
                curr = curr.children[1]
        return curr.leaf_class

def dumb_predictors(Y_test , message):
    print(f'Testing on {message}')
    print(f'Only win prediction accuracy {100 * np.sum(Y_test) / len(Y_test):.3f}%')
    print(f'Only loss prediction accuracy {100 * (1 -  np.sum(Y_test) / len(Y_test)):.3f}%\n')




def run_Problem_a():
 
    print('\n___Scratch Decision Tree______________________________________________________________________')
    for _max_depth_ in __MAX_DEPTHS__:
        tree = Decision_Tree()
        tree.fit(_cat_label_cnt_, X_train_cat, X_train_cnt, Y_train, _max_depth_)
        train_accuracies_scratch_pre.append(tree.test_accuracy(X_train_cat , X_train_cnt , Y_train , "Training data"))
        val_accuracies_scratch_pre.append(tree.test_accuracy(X_val_cat , X_val_cnt , Y_val , "Validation data"))
        test_accuracies_scratch_pre.append(tree.test_accuracy(X_test_cat , X_test_cnt , Y_test , "Testing data"))
        print()
    
    Ys = [train_accuracies_scratch_pre ,  val_accuracies_scratch_pre , test_accuracies_scratch_pre]
    Ylabels = ["trainng set" , "validation set" , "test set"]
    make_save_plot(__MAX_DEPTHS__ , 'Max depth of decision tree',  Ys ,  Ylabels , "Scratch accuracy pre-pruning" , "a_")



def write_lists(L1 , L2 , L3 , filename = "med.txt"):
    with open(filename, 'w') as file:
        for item in L1:
            file.write(str(item) + ' ')
        file.write('\n')  # Separate the lists with an empty line
        for item in L2:
            file.write(str(item) + ' ')
        file.write('\n')
        for item in L3:
            file.write(str(item) + ' ')

    
if __name__ == '__main__':
    label_encoder = None
    TRAIN_DATA_PATH = './data/train.csv'
    VAL_DATA_PATH = './data/val.csv'
    TEST_DATA_PATH = './data/test.csv'
    need_label_encoding = ['team', 'opp' , 'host', 'month', 'day_match']
    dont_need_label_encoding = ['year', 'toss', 'bat_first', 'format', 'fow', 'score', 'rpo', 'result']
    
    colours = ['blue' , 'red' , 'green' , 'magenta', 'black' , 'orange' , 'purple' ,'pink' , 'brown']

   
    _cat_label_cnt_, X_train_cat, X_train_cnt, Y_train = get_np_array(TRAIN_DATA_PATH , "Training")
    garbage, X_val_cat, X_val_cnt, Y_val = get_np_array(VAL_DATA_PATH , "Validation")
    garbage, X_test_cat, X_test_cnt, Y_test = get_np_array(TEST_DATA_PATH , "Testing")
    __MAX_DEPTHS__ = [i  for i in range(46)]


    ####################################################################################################################

    print("\n___Dumb Predictors______________________________________________________________________")
    dumb_predictors(Y_train , "training data")
    dumb_predictors(Y_val , "validation data")
    dumb_predictors(Y_test , "test data")

    ###(a)###############################################################################################################
    train_accuracies_scratch_pre = []
    val_accuracies_scratch_pre = []
    test_accuracies_scratch_pre = []
    run_Problem_a()
    write_lists(train_accuracies_scratch_pre , val_accuracies_scratch_pre , test_accuracies_scratch_pre)

    