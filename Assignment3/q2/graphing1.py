#!/usr/bin/env python3
import os
import numpy as np

# Open the text file for reading
import matplotlib.pyplot as plt

# layers = [1, 5, 10, 50, 100]
layers_arch = [ [512] , [512, 256] ,   [512, 256, 128] , [512, 256, 128, 64]]
colours = ['blue' , 'red' , 'green' , 'orange' , 'purple' ,  'magenta', 'black'  ,'pink' , 'brown']

def make_save_plot(X ,   Ys , Ylabels ,  title, name , save_folder = 'plots'):
    global colours
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(len(Ys)):
        plt.plot(X, Ys[i] , marker = 'o', markersize = 0.5  , color = colours[i] ,  label = Ylabels[i]  , linewidth = 0.5)

    plt.xlabel("Number of epoches")
    plt.ylabel('Cost J_(avg)')
    plt.title(title)
    legend = plt.legend(loc = 'center right')
    for handle in legend.legend_handles:
        handle.set_markersize(5)  # Adjust the size as needed


    save_path = os.path.join(save_folder, f'{name}.png')
    plt.savefig(save_path , dpi = 1000)

    plt.close()

x_inv = [2 ,1 , 0, 3]
x_re_inv = [1,2,3,0]
def get_plot(x):
    print(f"Parsing: './dump_inv/dump_{x}.txt'")
    with open(f'./dump_inv/dump_{x}.txt', 'r') as file:
        lines = file.readlines()

    i  = 1
    cost = []
    iter = []
    print(lines[0])
    while i < len(lines):
        cost_line = lines[i].strip()
        cost_value = float(cost_line.split(":")[1].strip())
        cost.append(cost_value)
        i+= 1
        # Extract the Iter value
        iter_line = lines[i].strip()
        iter_value = int(iter_line.split(":")[1].strip())
        iter.append(iter_value)
        if iter[-1] == 500:
            break
        # if iter[-1] == 5000:
        #     break
        i += 2

    return cost , iter

def get_plot_2(x):
    print(f"Parsing: './dump_re_inv/dump_{x}.txt'")
    with open(f'./dump_re_inv/dump_{x}.txt', 'r') as file:
        lines = file.readlines()

    i  = 1
    cost = []
    iter = []
    print(lines[0])
    while i < len(lines):
        cost_line = lines[i].strip()
        cost_value = float(cost_line.split(":")[1].strip())
        cost.append(cost_value)
        i+= 1
        # Extract the Iter value
        iter_line = lines[i].strip()
        iter_value = int(iter_line.split(":")[1].strip())
        iter.append(iter_value)
        if iter[-1] == 500:
            break
        # if iter[-1] == 5000:
        #     break
        i += 2

    return cost , iter





iter_ = None
# print(100 * np.array(Ys))
Ylab = ["Sigmoid" , "ReLU"]
for i in range(4):
    Ys = []
    cost , iter_ = get_plot(x_inv[i])
    print("Singmod")
    print(cost[0] , cost[-1])
    print(iter_[0] , iter_[-1])
    print(len(cost))
    print(len(iter_))
    Ys.append(cost)
    cost , iter_ = get_plot_2(x_re_inv[i])
    print("ReLU")
    print(cost[0] , cost[-1])
    print(iter_[0] , iter_[-1])
    print(len(cost))
    print(len(iter_))
    Ys.append(cost)
    make_save_plot(iter_, Ys , Ylab , f"Sigmoid vs ReLU: Layers: {layers_arch[i]}" , f"de_cost_{i}")

    # Ylab.append(f'Hidden Layer Width {layers[i]}')
    # print(len(Ys[-1]))

# x = [1000 , 2000 , 3000, 4000 , 5000]

# make_save_plot(iter_, Ys , Ylab , "Testing Accuracy vs Epoch: Single Hidden Layer" , "b_graph_acc_test")

# python3 hello.py

# epoch = [1000 , 2000 , 3000 , 4000 , 5000 ]
# A_train = 100 * np.array([ 0.79  , 0.86 ,    0.90     , 0.97   , 0.99 ])
# A_test = 100 * np.array([0.79   , 0.81 , 0.82  ,   0.84 ,  0.84  ])

# print(A_train)
# print(A_test)

# Ys=  [A_train , A_test]

# Yl = ["Training Accuray" , "Testing Accuracy"]
# make_save_plot(epoch, Ys , Yl , "Training and Test accuracies while training [100 50]" , "a_graph_acc")








