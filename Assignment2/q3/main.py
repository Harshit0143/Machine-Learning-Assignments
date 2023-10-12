#!/usr/bin/env python3
import numpy as np
import sklearn
import PIL
from PIL import Image
import time
import cvxopt
from cvxopt import solvers , matrix 

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

import matplotlib.pyplot as plt



prediction_key = np.array(["Negative" , "Positive"])

def __read_image_data__(__PATH_IMAGE_DATA__ , resize_to = 16):
    all_images = []
    for filename in os.listdir(__PATH_IMAGE_DATA__):
        image = PIL.Image.open(os.path.join(__PATH_IMAGE_DATA__, filename))
        image = image.resize((resize_to, resize_to) , PIL.Image.Resampling.NEAREST)
        image = np.array(image , dtype = np.float64) / 255.0
        image = image.flatten()
        all_images.append(image)
    return all_images
def merge_input_matrices(classwise_data):
    X =  np.concatenate(classwise_data)
    true_classification =  np.concatenate([np.full(len(classwise_data[i]) , i , dtype = np.int64) for i in range (len(classwise_data))])
    return X , true_classification

def build_input_matrices( __CLASSWISE_TRAIN_DATA_PATHS__ , resize_to = 16):   
    return [__read_image_data__(class_k_data_path , resize_to)  for class_k_data_path in __CLASSWISE_TRAIN_DATA_PATHS__]


def gaussian_kernel(X , Y):
    global GAMMA
    P = np.sum(X ** 2, axis = 1)
    P = P[:, np.newaxis] + P - 2 * np.matmul(X , X.T)
    P = np.exp(-GAMMA * P)
    P *= np.outer(Y, Y)
    return P
    
    


class Binary_Image_classifier_svm: # gaussian only

    def __init__(self , classwise_data):
        print("Running scratch Binary SVM classifier:")
        assert len(classwise_data) == 2,"Binary_Image_classifier_svm neeeds 2 class input!"
        self.X , self.true_classification = merge_input_matrices(classwise_data)
        self.Y = np.concatenate([np.full(len(classwise_data[i]) , 2 * i - 1.0 , dtype = np.float64) for i in range (2)])
        self.XY = self.X * self.Y[: , None]
        self.__build_convex_optimisation_problem__()

    def b_gaussian(self):
        zero_error_minus = self.phi_wT_phi_xi[np.logical_and(self.Y == -1 , self.alpha < 0.8)]
        zero_error_plus =  self.phi_wT_phi_xi[np.logical_and(self.Y == 1 , self.alpha < 0.8)]
        return -(np.max(zero_error_minus) + np.min(zero_error_plus)) / 2
    

    def __build_convex_optimisation_problem__(self , C = 1):
        m = len(self.Y)
        P_np = gaussian_kernel(self.X, self.Y)
        P = matrix(P_np)
        q = matrix([-1.0] * m , tc = 'd')

        G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc = 'd')
        h = matrix(([0.0] * m +  [C] * m) , tc = 'd')
    

        A = matrix([[yi] for yi in self.Y] , tc = 'd')
        b = matrix([0.0] , tc = 'd')
        sol = solvers.qp(P , q , G , h , A  , b) 
        self.alpha = np.array(sol['x'])


        self.phi_wT_phi_xi = self.Y * np.sum(self.alpha * P_np , axis = 0)
        self.alpha = np.concatenate(self.alpha)

        self.b = self.b_gaussian()
  
    def __score_plus__(self , feature_vector):
        global GAMMA
        distance = self.X - feature_vector
        distance = np.exp(-GAMMA * np.sum(distance ** 2 , axis = 1))
        distance *= self.alpha
        distance *= self.Y
        distance = np.sum(distance)
        distance += self.b
        return distance

        


class Multi_Image_classifier_svm: ## gaussian only
    def __init__(self , classwise_data):
        print("Running scratch MultiClass SVM classifier:")
        self.class_cnt = len(classwise_data)
        self.pairwise_models = [[Binary_Image_classifier_svm([classwise_data[i] , classwise_data[j]]) for j in range(i)] for i in range(self.class_cnt)]

    def __make_prediction__(self , feature_vector):
        votes = np.zeros(self.class_cnt , dtype = np.int64)
        score = np.zeros(self.class_cnt , dtype = np.float64)


        for i in range(self.class_cnt):
            for j in range(i):
                score_plus = self.pairwise_models[i][j].__score_plus__(feature_vector)
                if (score_plus < 0):
                    votes[i] += 1
                else:
                    votes[j] += 1
                score[i] -= score_plus
                score[j] += score_plus

        L = list(zip(votes, score))
        max_vote_score = max(L)
        for i in range(len(L)):
            if (L[i] == max_vote_score):
                return i
        

    def test_accuracy(self , __TEST_X__ , __TEST_TRUE_CLASS__):
        all_predictions = np.apply_along_axis(self.__make_prediction__ , axis = 1, arr = __TEST_X__)
        accuracy = np.sum(all_predictions == __TEST_TRUE_CLASS__) / len(__TEST_TRUE_CLASS__)
        print(f"Accuracy using Scratch Model: {accuracy * 100:.3f} %")
        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(__TEST_TRUE_CLASS__ , all_predictions)
        print(conf_matrix)
        return all_predictions


      


        
def scratch_SVM_classifier_build(__CLASSWISE_TRAIN_DATA_PATHS__):
    classwise_train_data = build_input_matrices(__CLASSWISE_TRAIN_DATA_PATHS__)
    t_train = time.time()
    obj = Multi_Image_classifier_svm(classwise_train_data)
    t_train = time.time() - t_train
    return obj , t_train
def scratch_SVM_classifier_test_accuracy(model , __CLASSWISE_TEST_DATA_PATHS__ , message):
    print(message)
    classwise_test_data = build_input_matrices(__CLASSWISE_TEST_DATA_PATHS__)
    __TEST_X__ , __TEST_TRUE_CLASS__  = merge_input_matrices(classwise_test_data)
    print("Total test size:" , len(__TEST_TRUE_CLASS__))
    return model.test_accuracy(__TEST_X__ , __TEST_TRUE_CLASS__) 
    




def sklearn_SVM_classifier_gaussian( __CLASSWISE_TRAIN_DATA_PATHS__ , __CLASSWISE_TEST_DATA_PATHS__):
    print("Running scikit learn :: SVM classifier:")
    classwise_train_data = build_input_matrices(__CLASSWISE_TRAIN_DATA_PATHS__)
    __TRAIN_X__ , __TRAIN_TRUE_CLASS__  = merge_input_matrices(classwise_train_data)
    print("Train data size:" , len(__TRAIN_TRUE_CLASS__))
    
    t_train = time.time()
    svm_classifier = SVC(kernel = "rbf" , C = 1.0 , gamma = GAMMA)
    svm_classifier.fit(__TRAIN_X__ , __TRAIN_TRUE_CLASS__)
    t_train = time.time() - t_train

    

    classwise_test_data = build_input_matrices(__CLASSWISE_TEST_DATA_PATHS__)
    __TEST_X__ , __TEST_TRUE_CLASS__  = merge_input_matrices(classwise_test_data)
    Y_predicted = svm_classifier.predict(__TEST_X__)
    accuracy = accuracy_score(__TEST_TRUE_CLASS__, Y_predicted)
    print(f"Accuracy of SK learn classifier: {accuracy * 100:.3f} %")
    print("Test data size:" , len(__TEST_TRUE_CLASS__))
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(__TEST_TRUE_CLASS__ , Y_predicted)
    print(conf_matrix)
    return t_train , Y_predicted




def make_save_plot(C_vals , k_fold_accuracy , validation_accuracy  , save_name = "Accyracy_vs_C"):
    X_axis = np.log(C_vals)
    Y1 = 100 * k_fold_accuracy
    Y2 = 100 * validation_accuracy
    plt.plot(X_axis, Y1 , marker = 'o', color = "red" , label = "Accuracy averaged k-fold cross-validation")
    plt.plot(X_axis, Y2 , marker = 'o', color = "blue" , label = "Accurcy on Validation set")

    plt.xlabel("log(C) values")
    plt.ylabel("Accuracy, in %")
    plt.legend()

    plt.savefig(save_name + ".png")


def shuffle_X_Y(X , Y):
    perm = np.random.permutation(len(Y))
    return X[perm] , Y[perm]

def train_set_exclude_pth(X_split  , Y_split , p):
    k = len(X_split)
    X_train = np.concatenate([X_split[j] for j in range(k) if j != p])
    Y_train = np.concatenate([Y_split[j] for j in range(k) if j != p])
    return X_train , Y_train


def pth_fold(X_split , Y_split ,  p , _C_):
    X_train , Y_train = train_set_exclude_pth(X_split , Y_split , p)
    X_test = X_split[p]
    Y_test = Y_split[p]
    svm_classifier = SVC(kernel = "rbf" , C = _C_ , gamma = GAMMA)
    svm_classifier.fit(X_train , Y_train)
    Y_predicted = svm_classifier.predict(X_test)
    return accuracy_score(Y_test, Y_predicted)
    

def evaluate_k_fold_error(X_split , Y_split , C):
    k = len(X_split)
    mean_accuracy = 0
    for i in range(k):
        mean_accuracy += pth_fold(X_split , Y_split , i , C)
    return mean_accuracy / k

def k_fold_accuracies(__MULTICLASS_TRAIN_DATA_PATH__ , C_VALS):
    classwise_train_data = build_input_matrices(__MULTICLASS_TRAIN_DATA_PATH__)
    __TRAIN_X__ , __TRAIN_TRUE_CLASS__  = merge_input_matrices(classwise_train_data)
    __TRAIN_X__ , __TRAIN_TRUE_CLASS__ = shuffle_X_Y(__TRAIN_X__ , __TRAIN_TRUE_CLASS__)
    
    k = 5
    X_split = np.split(__TRAIN_X__ , k)
    Y_split = np.split(__TRAIN_TRUE_CLASS__, k)

    accuracy_list = []
    for C in C_VALS:
        print(f"Running k-fold for C: {C}")
        accuracy_list.append(evaluate_k_fold_error(X_split , Y_split , C))
    return np.array(accuracy_list , dtype = np.float64)

def validation_accuracies(__MULTICLASS_TRAIN_DATA_PATH__  , __MULTICLASS_TEST_DATA_PATH__ , C_VALS):
    classwise_train_data = build_input_matrices(__MULTICLASS_TRAIN_DATA_PATH__)
    X_train , Y_train  = merge_input_matrices(classwise_train_data)


    classwise_test_data = build_input_matrices(__MULTICLASS_TEST_DATA_PATH__)
    X_test ,Y_test  = merge_input_matrices(classwise_test_data)


    validation_accuracies_list = []
    for _C_ in C_VALS:
        svm_classifier = SVC(kernel = "rbf" , C = _C_ , gamma = GAMMA)
        svm_classifier.fit(X_train ,Y_train)
        Y_predicted = svm_classifier.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predicted)
        validation_accuracies_list.append(accuracy)

    
    return np.array(validation_accuracies_list , dtype = np.float64)


def save_image_from_array(image_normalised_flat_array , name , folder = "images"):
    image_normalised_flat_array *= 255.0
    image_matrix = image_normalised_flat_array.astype(np.uint8).reshape(16, 16, 3)
    image = Image.fromarray(image_matrix)
    image = image.resize((10 * image.width, 10 * image.height), Image.NEAREST)
    destination_folder = folder
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder) 
    image.save(os.path.join(folder, name + ".png"))

def save_k_wrong_predictions(X_test , Y_test , Y_predicted  , indices , suffix):
    mismatched = Y_predicted != Y_test
    Y_test_mismatched = Y_test[mismatched]
    Y_pred_mismatched = Y_predicted[mismatched]
    L = []
    for i in range (len(indices)):
        idx = indices[i]
        L.append((Y_test_mismatched[idx] , Y_pred_mismatched[idx]))
        save_image_from_array(X_test[idx] ,"mis_classified" + str(i) + "_" + suffix) 
    print(L)
    
def save_specifics(X_test , restriced , indices , prefix):
    special_images = X_test[restriced]
    for i in range (len(indices)):
        idx = indices[i]
        save_image_from_array(special_images[idx] , prefix + str(i)) 
    
    
    
    


def run_Problem3():

    CLASS_CNT = 6
    __MULTICLASS_TRAIN_DATA_PATH__ = ["./svm/train/" + str(i) for i in range(0 ,CLASS_CNT)]
    __MULTICLASS_VALIDATION_DATA_PATH__ = ["./svm/val/" + str(i) for i in range(0 ,CLASS_CNT)]



    ##(a , c)################################################################################################
    print("Running Multiclass SVM classifier built from scratch:")
    scratch_multiclass_model , t_train_scratch = scratch_SVM_classifier_build(__MULTICLASS_TRAIN_DATA_PATH__)
    print(f"Training time of Scratch {t_train_scratch:.2f} seconds")
    Y_predicted_scratch = scratch_SVM_classifier_test_accuracy(scratch_multiclass_model , __MULTICLASS_VALIDATION_DATA_PATH__ , "Showing Validation Set Accuracy")

    ##(b , c)################################################################################################
    print("Running SVM classifier from sk_learn:")
    t_train_sk_learn , Y_predicted_SK  = sklearn_SVM_classifier_gaussian(__MULTICLASS_TRAIN_DATA_PATH__ , __MULTICLASS_VALIDATION_DATA_PATH__)
    print(f"Training time of SK Learn {t_train_sk_learn:.2f} seconds")
    
    ## (c)################Pick out the random images
    classwise_test_data = build_input_matrices(__MULTICLASS_VALIDATION_DATA_PATH__)
    X_test , Y_test  = merge_input_matrices(classwise_test_data)
    
    boolean_4_as_2 = np.logical_and(Y_test == 4 ,Y_predicted_scratch == 2)
    boolean_2_as_2 = np.logical_and(Y_test == 2 ,Y_predicted_scratch == 2)
    indices1 = [51 , 38 , 48 , 52 , 41 , 3 , 39 , 14]
    indices2 = [9 , 29 , 103 , 87 , 37 , 114 , 109, 27]
    save_specifics(X_test , boolean_4_as_2 ,indices1 ,"_4_as_2_")
    save_specifics(X_test , boolean_2_as_2 ,indices2 ,"_2_as_2_")

    

    indices3 = [392 , 393 , 454 , 89 , 181 , 70]
    indices4 = [97 , 453 , 129 , 433 , 431 , 410]
    save_k_wrong_predictions(X_test , Y_test , Y_predicted_scratch , indices3 , "scratch" )
    save_k_wrong_predictions(X_test , Y_test , Y_predicted_SK , indices4 , "sk_learn" )




    ##(d)####################################################################################################
    print("Runnning k-fold cross validation!")
    C_VALS = np.array([1e-5 , 1e-3 , 1 , 5 , 10] , dtype = np.float64)
    k_fold_accuracies_ = k_fold_accuracies(__MULTICLASS_TRAIN_DATA_PATH__ , C_VALS)

    validation_accuracies_ = validation_accuracies(__MULTICLASS_TRAIN_DATA_PATH__ , __MULTICLASS_VALIDATION_DATA_PATH__ , C_VALS)
    print("K fold accuracies:" , k_fold_accuracies_)     
    print("Validation accuracies:" , validation_accuracies_)     
    make_save_plot(C_VALS , k_fold_accuracies_ , validation_accuracies_)




if __name__ == "__main__":
    GAMMA = 0.001
    EPSILON = 1e-6
    solvers.options["show_progress"] = False
    ##Problem 3:#########################################################################################   
    run_Problem3()



