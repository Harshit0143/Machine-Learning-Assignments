#!/usr/bin/env python3
import os
import numpy as np
import sys

# Open the text file for reading
import matplotlib.pyplot as plt

layers = [1, 5, 10, 50, 100]
layers_arch = [ [512] , [512, 256] ,   [512, 256, 128] , [512, 256, 128, 64]]
colours = ['blue' , 'red' , 'green' , 'orange' , 'purple' ,  'magenta', 'black'  ,'pink' , 'brown']

def make_save_plot(X ,   Ys , Ylabels ,  title, name , save_folder = 'plots'):
    global colours
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(len(Ys)):
        plt.plot(X, Ys[i] , marker = 'o', markersize = 4  , color = colours[i] ,  label = Ylabels[i]  , linewidth = 1)

    plt.xlabel("Network Depth")
    plt.ylabel('F1 Score (%)')
    plt.title(title)
    legend = plt.legend(loc = 'center right')
    for handle in legend.legend_handles:
        handle.set_markersize(5)  # Adjust the size as needed


# 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    save_path = os.path.join(save_folder, f'{name}.png')
    plt.savefig(save_path , dpi = 1000)

    plt.close()











def read_data(filename):
    print(f"Parsing: {filename}'")
    with open(filename, 'r') as file:
        lines = file.readlines()
# 14 27 45 58 76 89 
    i  = 1
    f1_train = []
    f1_test = []
    prec_train = []
    prec_test = []
    rec_test =[]
    rec_train = []
    epoches = []
    print(lines[0])
    train = True
    while i < len(lines):
        line = lines[i].strip().split()
        if 'epoch:' in line:
            epoches.append(int(line[1]))
        elif ('weighted' in line) and train:
            prec_train.append(100 * float(line[2]))
            rec_train.append(100 * float(line[3]))
            f1_train.append(100 * float(line[4]))

            train = False

        elif ('weighted' in line):
            prec_test.append(100 * float(line[2]))
            rec_test.append(100 * float(line[3]))
            f1_test.append(100 * float(line[4]))
            train = True

        i += 1

    return prec_train , prec_test , rec_train , rec_test , f1_train , f1_test , epoches

Ys_train = []
Ys_test =  []
p_train = []
p_test = []
r_train = []
r_test = []
Y_lab = []
# for i in range(4):
#     prec_train , prec_test , rec_train , rec_test , f1_train , f1_test , epoch= read_data(f"./acc_re_inv/acc_{i}.txt")
#     Ys_train.append(f1_train)
#     p_test.append(prec_test)
#     p_train.append(prec_train)
#     r_train.append(rec_train)
#     r_test.append(rec_test)
#     Ys_test.append(f1_test) 
#     Y_lab.append(f'Hidden Layers: {layers_arch[i]}')
#     print(f"Len: {len(f1_train)}")
#     print("Training")
#     print(f1_train[0] , f1_train[-1])
#     print("Testing")
#     print(f1_test[0] , f1_test[-1])


# print(epoch)

train =   np.array([[0.63402 ,  0.59240  , 0.60704],
        [0.63245,   0.61480 ,  0.62189],
[0.63908  , 0.62880  , 0.63326],
[ 0.65539 ,  0.64570 ,  0.64980 ]])


test = np.array([[ 0.62355  , 0.58700 ,  0.59947],
  [  0.62361  , 0.61300 ,  0.61709] ,
   [ 0.63413  , 0.63100 ,  0.63186],
    [0.64720,   0.64500  , 0.64603]])

for i in range(4):  
    train[i] *= 100
    
    print(f"|${layers_arch[i]}$ |${train[i][0]:.3f}$|${train[i][1]:.3f}$|${train[i][2]:.3f}$|")

for i in range(4):  
    test[i] *= 100
    
    print(f"|${layers_arch[i]}$ |${test[i][0]:.3f}$|${test[i][1]:.3f}$|${test[i][2]:.3f}$|")

f1_train = train[: , 2]
f1_test = test[: , 2]
Ys = [f1_train , f1_test]
Y_lab = ["Training" , "Testing"]
make_save_plot([i for i in range(4)], Ys , Y_lab , f"F1 Score vs Network Depth: SK Learn" , f"f_scores_relative")

# for i in range(4):
    # print(f"|${layers_arch[i]}$ |${p_train[i][9]:.3f}$|${r_train[i][9]:.3f}$|${Ys_train[i][9]:.3f}$|")


sys.exit()

print()
for i in range(4):
    print(f"|${layers_arch[i]}$ |${p_test[i][9]:.3f}$|${r_test[i][9]:.3f}$|${Ys_test[i][9]:.3f}$|")
Ys = [[] ,[]]
for i in range(4):
    Ys[0].append(Ys_train[i][9])
for i in range(4):
    Ys[1].append(Ys_test[i][9])
    
print(Ys[0])
print(Ys[1])
# make_save_plot(epoch, Ys_test , Y_lab , f"F1 Score vs Epoches: Test" , f"c_scores_test")

# for i in range(len(f1_test)):
#     print(f'|${epoch[i]}$|${f1_train[i]:.3f}$|${f1_test[i]:.3f}$|')

# 0.64459   0.58280   0.60327
# 0.62517   0.59600   0.60699
# 0.63722   0.61730   0.62518



# 0.61399   0.55800   0.57621
# 0.60897   0.58700   0.59506    
# 0.62542   0.61300   0.61779 