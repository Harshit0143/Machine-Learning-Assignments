#!/usr/bin/env python3
import numpy as np
import sklearn
import PIL
from PIL import Image
import time
import cvxopt
from cvxopt import solvers , matrix 

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
    
    

def linear_kernel(X , Y):    
    XY = X * Y[: , None]
    return np.matmul(XY , XY.T)

def save_image_from_array(image_normalised_flat_array , name , folder = "images"):
    image_normalised_flat_array *= 255.0
    image_matrix = image_normalised_flat_array.astype(np.uint8).reshape(16, 16, 3)
    image = Image.fromarray(image_matrix)
    image = image.resize((10 * image.width, 10 * image.height), Image.NEAREST)

    destination_folder = folder
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder) 
    image.save(os.path.join(folder, name + ".png"))

def show_common_cnt(arr1 , arr2, messages):
    common_indices = np.sum(np.logical_and(arr1 , arr2))
    print(messages , common_indices)

class Binary_Image_classifier_svm:

    def __init__(self , classwise_data , kernel = "linear"):
        print("Running self deisgned Binary SVM classifier:")
        assert len(classwise_data) == 2,"Binary_Image_classifier_svm neeeds 2 class input!"
        self.X , self.true_classification = merge_input_matrices(classwise_data)
        self.Y = np.concatenate([np.full(len(classwise_data[i]) , 2 * i - 1.0 , dtype = np.float64) for i in range (2)])
        self.XY = self.X * self.Y[: , None]
        self.type = kernel
        self.__build_convex_optimisation_problem__()

    def show_support_vector_cnt(self , epsilon):
        is_sv_bool = self.alpha > epsilon
        sv_cnt = np.sum(is_sv_bool)
        print("epsilon" , epsilon)
        print("Number of Support Vectors:" , sv_cnt)
        sv_percentage = (sv_cnt * 100) / len(self.alpha)
        print(f"Percentage of training samples that are support vectors: {sv_percentage:.3f} %") 
        return is_sv_bool 
    
    def show_model_parameters(self):
        print("Final values found:")
        np.set_printoptions(precision = 3)
        print("w:" ,self.w)
        print(f"b:{self.b:.3f}")
        np.set_printoptions()

    def save_top_k_support_vectors(self , k , suffix):
        top_indices = np.argsort(self.alpha)[-k:]  
        top_k_support_vectors = self.X[top_indices]
        print(f"Top {k} support vectors:")
        print(self.alpha[top_indices])
        for i in range(k):
            save_image_from_array(top_k_support_vectors[i] , "top_sv_" + suffix + '_' + str(i + 1))


    def b_gaussian(self):
        zero_error_minus = self.phi_wT_phi_xi[np.logical_and(self.Y == -1 , self.alpha < 0.8)]
        zero_error_plus =  self.phi_wT_phi_xi[np.logical_and(self.Y == 1 , self.alpha < 0.8)]
        return -(np.max(zero_error_minus) + np.min(zero_error_plus)) / 2
    
    def b_linear(self):
        b_arr = self.Y - np.matmul(self.X , self.w)
        b_contribute = b_arr[np.logical_and(self.alpha > 0.1 , self.alpha < 0.9)]
        return np.sum(b_contribute) / len(b_contribute)

         

    def __build_convex_optimisation_problem__(self, C = 1.0):
        m = len(self.Y)
        P_np = linear_kernel(self.X , self.Y) if (self.type == "linear") else gaussian_kernel(self.X, self.Y)
        P = matrix(P_np)
        q = matrix([-1.0] * m , tc = 'd')

        G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc = 'd')
        h = matrix(([0.0] * m +  [C] * m) , tc = 'd')
    

        A = matrix([[yi] for yi in self.Y] , tc = 'd')
        b = matrix([0.0] , tc = 'd')
        sol = solvers.qp(P , q , G , h , A  , b) 
        self.alpha = np.array(sol['x'])



        self.w = np.sum(self.alpha * self.XY , axis = 0)
        self.phi_wT_phi_xi = self.Y * np.sum(self.alpha * P_np , axis = 0)
        self.alpha = np.concatenate(self.alpha)

        self.b = self.b_linear() if self.type == "linear" else self.b_gaussian()
        
       

    
    def __make_prediction_linear__(self , feature_vector):
        distance = self.b + np.sum(self.w * feature_vector)
        return 1 if distance > 0 else 0   

    
    def __make_prediction_gaussian__(self , feature_vector):
        global GAMMA
        distance = self.X - feature_vector
        distance = np.exp(-GAMMA * np.sum(distance ** 2 , axis = 1))
        distance *= self.alpha
        distance *= self.Y
        distance = np.sum(distance)
       
        distance += self.b
        return 1 if distance > 0 else 0 
    
    def test_accuracy(self , __TEST_X__ , __TEST_TRUE_CLASS__):
        print("Test set size:" , len(__TEST_TRUE_CLASS__))
        if (self.type == "linear"):
            all_predictions = np.apply_along_axis(self.__make_prediction_linear__ , axis = 1 , arr = __TEST_X__)
        else:
            all_predictions = np.apply_along_axis(self.__make_prediction_gaussian__ , axis = 1 , arr = __TEST_X__)

        accuracy = np.sum(all_predictions == __TEST_TRUE_CLASS__) / len(__TEST_TRUE_CLASS__)
        print(f"Accuracy: {accuracy * 100:.3f} %")
       

        




def scratch_SVM_classifier_test_accuracy(model , __CLASSWISE_TEST_DATA_PATHS__ , message):
    print(message)
    classwise_test_data = build_input_matrices(__CLASSWISE_TEST_DATA_PATHS__)
    __TEST_X__ , __TEST_TRUE_CLASS__  = merge_input_matrices(classwise_test_data)
    model.test_accuracy(__TEST_X__ , __TEST_TRUE_CLASS__)

def sklearn_SVM_classifier_linear(__CLASSWISE_TRAIN_DATA_PATHS__ , __CLASSWISE_TEST_DATA_PATHS__):
    print("Running scikit learn :: SVM classifier:")
    classwise_train_data = build_input_matrices(__CLASSWISE_TRAIN_DATA_PATHS__)
    __TRAIN_X__ , __TRAIN_TRUE_CLASS__  = merge_input_matrices(classwise_train_data)
    t_train  = time.time()
    svm_classifier = SVC(kernel = "linear" , C = 1.0 , random_state = 42)
    svm_classifier.fit(__TRAIN_X__ , __TRAIN_TRUE_CLASS__)
    t_train = time.time() - t_train
    classwise_test_data = build_input_matrices(__CLASSWISE_TEST_DATA_PATHS__)
    __TEST_X__ , __TEST_TRUE_CLASS__  = merge_input_matrices(classwise_test_data)
    Y_predicted = svm_classifier.predict(__TEST_X__)
    accuracy = accuracy_score(__TEST_TRUE_CLASS__, Y_predicted)
    print(f"Accuracy of SK_Learn Linear: {accuracy * 100:.3f} %")

    return svm_classifier.coef_ , svm_classifier.intercept_[0] ,indices_to_boolean(svm_classifier.support_, len(__TRAIN_X__)) , t_train


def sklearn_SVM_classifier_gaussian( __CLASSWISE_TRAIN_DATA_PATHS__ , __CLASSWISE_TEST_DATA_PATHS__):
    print("Running scikit learn :: SVM classifier:")
    classwise_train_data = build_input_matrices(__CLASSWISE_TRAIN_DATA_PATHS__)
    __TRAIN_X__ , __TRAIN_TRUE_CLASS__  = merge_input_matrices(classwise_train_data)
    t_train  = time.time()
    svm_classifier = SVC(kernel = "rbf" , C = 1.0 , gamma = GAMMA)
    svm_classifier.fit(__TRAIN_X__ , __TRAIN_TRUE_CLASS__)
    t_train = time.time() - t_train
    
    
    classwise_test_data = build_input_matrices(__CLASSWISE_TEST_DATA_PATHS__)
    __TEST_X__ , __TEST_TRUE_CLASS__  = merge_input_matrices(classwise_test_data)
    print("Test set size:" , len(__TEST_TRUE_CLASS__))
    Y_predicted = svm_classifier.predict(__TEST_X__)
    accuracy = accuracy_score(__TEST_TRUE_CLASS__, Y_predicted)
    print(f"Accuracy of SK_Learn Gaussian: {accuracy * 100:.3f} %")
    return indices_to_boolean(svm_classifier.support_ , len(__TRAIN_X__)) , t_train



def show_train_times(scratch_linear , scratch_gaussian , sk_linear , sk_gaussian):
    print("Training times:")
    print(f"scratch  :: linear kernel  : {scratch_linear:.3f} seconds")
    print(f"scratch  :: gaussian kernel: {scratch_gaussian:.3f} seconds")
    print(f"sk learn :: linear kernel : {sk_linear:.3f} seconds" )
    print(f"sk learn :: gaussian kernel: {sk_gaussian:.3f} seconds")

def degree_angle_between_lines(w1 , w2):
    dot_val = np.sum(w1 * w2)
    mod1 = np.linalg.norm(w1)
    mod2 = np.linalg.norm(w2)
    cosine_theta = dot_val / (mod1  * mod2)
    angle_degrees = np.degrees(np.arccos(cosine_theta))
    return angle_degrees



def compare_classifiers(name1 , w1 , b1, name2 , w2 , b2):
    print("b values:")
    print(f"b_{name1}: {b1:.3f}")
    print(f"b_{name2}: {b2:.3f}")
    angle = degree_angle_between_lines(w1 , w2)
    print(f"Angle between w_{name1} and w_{name2} in degrees: {angle:.3f}")

def indices_to_boolean(support_vector_indices , size):
    is_support_vector = np.zeros(size, dtype = int)
    is_support_vector[support_vector_indices] = 1
    return is_support_vector


    
def run_Problem2():
    __CLASSWISE_TRAIN_DATA_PATHS__ = ["./svm/train/" + str(i) for i in range(3 ,5)]
    __CLASSWISE_VALIDATION_DATA_PATHS__ = ["./svm/val/" + str(i) for i in range(3 ,5)]
    classwise_train_data = build_input_matrices(__CLASSWISE_TRAIN_DATA_PATHS__)
    ##(a)################################################################################################
    print("Using data:" , __CLASSWISE_TRAIN_DATA_PATHS__)
    print("______________Linear Kernel Model______________")
    time_scratch_linear = time.time()
    model_binary_LINEAR = Binary_Image_classifier_svm(classwise_train_data , kernel = "linear")
    time_scratch_linear = time.time() - time_scratch_linear
    isSV_linear = model_binary_LINEAR.show_support_vector_cnt(EPSILON)
    scratch_SVM_classifier_test_accuracy(model_binary_LINEAR , __CLASSWISE_VALIDATION_DATA_PATHS__ , message = "Showing Validation Set Accuracy:")
    save_image_from_array(model_binary_LINEAR.w , "w_image_linear")
    model_binary_LINEAR.save_top_k_support_vectors(6 , suffix = "linear")


    ##(b)################################################################################################
    print("______________Gaussian Kernel Model______________")
    time_scratch_gaussian = time.time()
    model_binary_GAUSSIAN = Binary_Image_classifier_svm(classwise_train_data , kernel = "gaussian")
    time_scratch_gaussian = time.time() - time_scratch_gaussian
    isSV_gaussian = model_binary_GAUSSIAN.show_support_vector_cnt(EPSILON)
    show_common_cnt(isSV_linear , isSV_gaussian ,"Number of SVs common in Linear and Gaussian Kernel:")
    scratch_SVM_classifier_test_accuracy(model_binary_GAUSSIAN , __CLASSWISE_VALIDATION_DATA_PATHS__ , message = "Showing Validation Set Accuracy:")
    save_image_from_array(model_binary_LINEAR.w , "w_image_gaussian")
    model_binary_GAUSSIAN.save_top_k_support_vectors(6 , suffix = "gaussian")
  
    ##(c)################################################################################################
    print("______________SK learn Linear Kernel______________")
    sk_linear_w , sk_linear_b , isSV_linear_sk ,time_sk_linear = sklearn_SVM_classifier_linear(__CLASSWISE_TRAIN_DATA_PATHS__ , __CLASSWISE_VALIDATION_DATA_PATHS__)
    print("______________SK learn Gaussian Kernel______________")
    isSV_gaussian_sk , time_sk_gaussian = sklearn_SVM_classifier_gaussian(__CLASSWISE_TRAIN_DATA_PATHS__ , __CLASSWISE_VALIDATION_DATA_PATHS__)
    show_train_times(time_scratch_linear , time_scratch_gaussian , time_sk_linear , time_sk_gaussian)
    

    show_common_cnt(isSV_linear_sk , isSV_gaussian_sk ,"Number of SVs common in SK learn's Linear and Gaussian Kernel:")
    compare_classifiers("sk_linear" , sk_linear_w , sk_linear_b , "scratch_linear" , model_binary_LINEAR.w , model_binary_LINEAR.b)
    
    
    print("Number of Support Vectors in SK_learn_linear" , np.sum(isSV_linear_sk))
    show_common_cnt(isSV_linear , isSV_linear_sk ,"Number of SVs common in scratch_linear and SK_Linear:")
    print("Number of Support Vectors in SK_learn_gaussian" , np.sum(isSV_gaussian_sk))
    show_common_cnt(isSV_gaussian , isSV_gaussian_sk ,"Number of SVs common in scratch_gaussian and SK_gaussian:")
    





if __name__ == "__main__":
    GAMMA = 0.001
    EPSILON = 1e-6
    solvers.options["show_progress"] = False
    run_Problem2()

    



