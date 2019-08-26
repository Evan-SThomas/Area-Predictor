import shutil
import os
file_path = os.path.realpath(__file__)
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import sys
import h5py
#sys.path.append(os.path.abspath("/home/et/pytt/geo/atools/"))
sys.path.append(os.path.abspath("/home/et/atools/"))
import libscoreh5
import libdupl
from itertools import islice
device  = torch.device("cuda")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Conv2d(1, 6, 2) #kernal size is 5.. 
#                                        so might it be larger than the smallest resolable feature?
        self.pool = nn.MaxPool2d(2,stride=1)#, 2) #output of conv is 19**2. mod(2) is 9
        self.conv2 = nn.Conv2d(6, 1, 2)
        self.fc1 = nn.Linear(16*16,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def absl(o,e):
    return abs(o-e)

net = Net()

AvgErr_v_TA = []
Avgpred_v_TA = []
NT_v_TA = []
RelErr_v_TA = []
slt_bw = []

#data_dir = "/home/et/pytt/geo/a_hash_det/"
#res_dir  = "/home/et/pytt/geo/a_hash_det/results/"
data_dir = "/home/et/ahash"
res_dir  = "/home/et/ahash/results/"

os.chdir(data_dir)
epoch_mult = 20 #Number of epochs in each cross-validation training run
batch_mult = 20 #Number of Batches in an epoch
max_area = 200 #Max area used to normalize data to within 1
running_loss = 0.0 #initialize

with h5py.File("td_hash.hdf5",'r') as hf:
    #Making Results Directory
    os.chdir(res_dir)
#    print(file_path)
    result_class = file_path.split("/")[-1][:-3] #first: what file are results from
    result_number = str(len(os.listdir())) #random counter
    os.mkdir(str(result_class+"_"+result_number))
    os.chdir(str(result_class+"_"+result_number))
    dest_path = str(os.getcwd()+"/"+result_class+".py")
#    print(dest_path)
    shutil.copyfile(file_path,dest_path)
    #Cross-Validation Groups. Currently non-general.
    hash_list= ['0','1','2','3']
    hg_lcs = [] #Hash-Group, Learning Curves

    #Loading Data for the epoch 
    key_list_master = [list(hf['t'][y].keys()) for y in hash_list]
    area_list_master = [[hf['t'][hash_list[y]][x].attrs['area'] for x in key_list_master[y]] for y in range(len(hash_list))]

    #Balencing data:
        #key_list is defined twice so that there are an even number of each class in each hashgroup. 
    key_list_master = [libdupl.dupl(key_list_master[x],area_list_master[x]) for x in range(len(key_list_master))]
    
    for cross_train_index in range(len(hash_list)):
        net = 0
        net = nn.DataParallel(Net())
        net.to(device)
        optimizer = optim.SGD(net.parameters(),lr=0.00125,momentum=0.5)

        #Specify which hash group is for testing
        train_list = hash_list[:] 
        test_list = train_list.pop(cross_train_index) #leaves train_list w/ len = 3

        #kl - Key List. Specificies the name of each data, in each group used for training.
        kl = [key_list_master[hash_list.index(x)] for x in train_list] #kl lists /hashgroup/names
        kl_test = key_list_master[hash_list.index(test_list)]

        train_prog_list = []
        lcl = [] #Learning Curve list
        print("\nTest Group: ",test_list)
        epoch_time = 0
        for epoch in range(1,epoch_mult):
            tb = time.time()
            print("epoch: ",epoch,"\ttrain group: ",train_list[epoch%3],"\ttime: ",epoch_time)
            epoch_size = len(kl[epoch%3]) #training each epoch from diff xvald. set, epoch size changes
            batch_size = int(epoch_size/batch_mult) #so does batch_size

            for batch_index in range (1,batch_mult):  #Loop over the batch multiple times
                print("batch_index: ",batch_index,"\tavg loss: ",running_loss / batch_size) 
                running_loss = 0.0 #Reset the loss for the new batch
                batch = [kl[epoch%3][x] for x in np.random.randint(epoch_size,size=batch_size)] 
                # Each batch is a random selection from the epoch's train group.

                for train_index in range(batch_size):
                    try:
                        optimizer.zero_grad()
                        #Defining input
                        tfile = hf['t'][train_list[epoch%3]][batch[train_index]]
                        inputs = torch.cuda.FloatTensor(tfile[:]).reshape((1,1,20,20)) 
                        dev_inputs = inputs.to(device)
                        #Forwards prop.
                        outputs = net(dev_inputs)
                        #Calculate loss and Backwards prop.
                        loss = absl(outputs[0][0],tfile.attrs['area']/max_area)
                        loss.backward() #Not best: loss is backprop'd after each fwrd prop.
                        optimizer.step()
                        running_loss += loss.item()

                    except Exception as e:
                        print(e)
            epoch_time = time.time()-tb
            lcl.append([epoch,running_loss/batch_size])
            train_prog_list.append(str('[%d, %d] loss: %.3f' %
                              (epoch , batch_index , running_loss / batch_size))) 
            print("Pred: ",float(outputs[0][0]),"\tTrue: ",tfile.attrs['area']/max_area,"\tloss: ",running_loss/batch_size)


        lcl = np.array(lcl)
        #Writing training progress to log
        train_prog = open(str(result_class+"_"+result_number)+".txt",'a')
        train_prog.write("\nTrain Group "+test_list+"\n")
        for i in train_prog_list:
            train_prog.write(str(i)+"\n")
        train_prog.close()

        #Save net from cross-validation run to file
        saved_net_name = str(result_class+"_"+str(result_number)+"_tg"+test_list+".pt")
        torch.save(net.state_dict(),saved_net_name)
        
        #Scoring
        score = libscoreh5.score(kl_test,net,hf['t'][test_list],max_area)
        #Writing Scoring to File
        scoredoc = open(str(result_class+"_score"+result_number+".txt"),"a")
        scoredoc.write("\nTrain Group "+test_list+"\n")
        scoredoc.write("area\t\texamples\t\tnrm'd score\t\t\tavg prediction\t\trelative error\n")
        try: #Check if 0 area triangle is in this test-group. If so, add '~' for its relative erro.
            zidum = score[0].index(0)
            scoredum = [x[:] for x in score[:]]
            scoredum[4].insert(zidum,'~') # adds a '-' into the place for relative score of zero.
            scorewrite = scoredum[:]
        except: #Except if zero is not in test group.
            scorewrite = score[:]
        #Writing to the file
        for j in list(range(len(score[0]))):
            for i in scorewrite:
                try:
                    scoredoc.write(str(i[j]))
                    scoredoc.write("\t\t")
                except:
                    scoredoc.write("-")
                    scoredoc.write("\t\t")
            scoredoc.write("\n")
        scoredoc.close()

        #Sort performance metrics
        slt = score[0]
        nlt = score[1]
        nrmdsc = score[2]
        rel_err = score[3]
        #Save performance metrics for cross-validation plots.
        slt_bw.append(np.array(slt))
        AvgErr_v_TA.append(nrmdsc)
        Avgpred_v_TA.append(score[3])
        NT_v_TA.append(nlt)
        RelErr_v_TA.append(rel_err)
        lcl = np.array(lcl)
        hg_lcs.append(lcl)
    


    #Plotting
    bar_width = min(abs(np.diff(slt)))/2
    plt.figure()
    for i,o in enumerate(AvgErr_v_TA):
        plt.scatter(slt_bw[i],o)
        #plt.bar(slt_bw[i]+bar_width*(-0.5+i/len(AvgErr_v_TA)),o,width=bar_width/(1.25*len(hash_list)),align='center',edgecolor='k')
    plt.legend(hash_list)
    plt.title("Train/Test: Average Error by Triangle Area")
    plt.xlabel("Trinagle Area")
    plt.ylabel("Average Error")
    plt.savefig('AverageError_v_TriangleArea_TG.png')
    plt.close()
    
    plt.figure()
    for i,o in enumerate(NT_v_TA):
        plt.scatter(slt_bw[i],o)
        #plt.bar(slt_bw[i]+bar_width*(-0.5+i/len(NT_v_TA)),o,width=bar_width/(1.25*len(hash_list)),align='center',edgecolor='k')
    plt.legend(hash_list)
    plt.title("Train/Test: Number of Tests by Triangle Area")
    plt.xlabel("Triangle Area")
    plt.ylabel("Number of Tests")
    plt.savefig('NumberofTest_v_TriangleArea_TG.png')
    plt.close()

    plt.figure()
    for i,o in enumerate(RelErr_v_TA):
        plt.scatter(slt_bw[i],o)
        #plt.bar(slt_bw[i]+bar_width*(-0.5+i/len(RelErr_v_TA)),o,width=bar_width/(1.25*len(hash_list)),align='center',edgecolor='k')
    plt.legend(hash_list)
    plt.title("Relative Average Error by Triangle Area")
    plt.xlabel("Triangle Area")
    plt.ylabel("Relative Average Error (%)")
    plt.savefig('RelativeAverageError_v_TriangleArea_TG.png')
    plt.close()

    plt.figure()
    for i,o in enumerate(Avgpred_v_TA):
        plt.scatter(slt_bw[i],o)
        #plt.bar(slt_bw[i]+bar_width*(-0.5+i/len(Avgpred_v_TA)),o,width=bar_width/(1.25*len(hash_list)),align='center',edgecolor='k')
    plt.legend(hash_list)
    plt.title("Average Prediction by Triangle Area")
    plt.xlabel("Trinagle Area")
    plt.ylabel("Average Prediction")
    plt.savefig("AveragePrediction_v_TriangleArea_TG.png")
    plt.close()

    plt.figure()
    for i in hg_lcs:
        plt.plot(i[:,0],i[:,1])
    plt.legend(hash_list)
    plt.title("Cross-Validation Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend(hash_list)
    plt.savefig("Cross-Validation_LearningCurves.png")
    plt.close()
    
    try:
        plt.figure()
        for i in hg_lcs:
            plt.semilogy(i[:,0],i[:,1])
        plt.title("Cross-Validation Log10 Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.legend(hash_list)
        plt.savefig("Cross-Validation_Log10LearningCurves.png")
        plt.close()

    except:
        plt.figure() 
        for i in hg_lcs:
            plt.semilogy(i[1:][:,0],i[:1][:,1])
        plt.title("Cross-Validation Log10 Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.legend(hash_list)
        plt.savefig("Cross-Validation_Log10LearningCurves.png")
        plt.close()
