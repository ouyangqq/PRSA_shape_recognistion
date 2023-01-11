# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:52:42 2022

@author: qiang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:16:20 2021
@author: qiang
"""
import sys
import time
import numpy as np
import pandas as pd 
import warnings
from torch.utils.data.sampler import WeightedRandomSampler
from torch import optim 
import torch
import random
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
#import ultils as alt
from sklearn.metrics import roc_curve, auc

import metrics as mc
import Model_Metrics as mmc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ldt=np.load('gdata/shape_for_trianing_ANN.npy')
[Train,Valdt]=ldt.item()[0],ldt.item()[0]
[Train_lb,Valdt_lb]=ldt.item()[1],ldt.item()[1]
fg=Train.shape[1]-9
selnum=len(Train)
X=torch.from_numpy(Train.astype(np.float32)).to(device)
Y=torch.from_numpy(Valdt_lb.astype(np.float32)).to(device)
X_test=torch.from_numpy(Train.astype(np.float32)).to(device)
Y_test=torch.from_numpy(Valdt_lb.astype(np.float32)).to(device)



#注意这里hid_dim 设置是超参数(如果太小，效果就不好)，使用tanh还是relu效果也不同，优化器自选
hid_dim_1 =16
hid_dim_2 =32
hid_dim_3 =512

d = X.shape[1]
d_out = Y.shape[1]
classnum=d_out


net = nn.Sequential(
                     nn.Linear(d,hid_dim_1),
                     nn.ReLU(), #核函数
                     nn.Linear(hid_dim_1, d_out)
                     ).to(device)

ms=str(hid_dim_1)+'_'+str(hid_dim_2)+'_'+str(hid_dim_3)
epochs=1000
alaph=0.01
Batchsize=1000


def Train_model():
    print("OPTIMIZER = optim.Adam(model.parameters(),lr = 0.001) \n ")
    print('Functions Ready')
    #定义损失函数与优化器0
    criterion = nn.MSELoss() 
    optimizer =torch.optim.Adam(net.parameters(), alaph)
    start = time.time()
    #dropout=0
    loss_accbuf=[]
    for epoch in range(epochs):
        t1=time.time()
        running_loss = 0
        step = 0

        rands=random.sample(range(0,selnum),selnum)  
        for i in range(int(selnum/Batchsize)+1):
            step += 1
            rand_sels = rands[Batchsize*i:Batchsize*(i+1)]

            buf=Y[rand_sels]

            label = torch.reshape(buf,(buf.shape[0],buf.shape[1]))
            optimizer.zero_grad()
            
            buf=X[rand_sels,:]

            x = torch.reshape(buf,(buf.shape[0],buf.shape[1]))
            outputs = net(x)
            
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            max_vals,Truths1= torch.max(Y[rand_sels,0:classnum],1)
            acc1=mc.calc_accuracy(outputs[:,0:classnum],Truths1)
            

        
        running_loss1=0 
        for i in range(int(Y_test.shape[0]/Batchsize)+1):
            buf=Y_test[Batchsize*i:Batchsize*(i+1)]
            #print(i,buf.shape)
            label1 = torch.reshape(buf,(buf.shape[0],buf.shape[1]))
            buf=X_test[Batchsize*i:Batchsize*(i+1),:]
            x = torch.reshape(buf,(buf.shape[0],buf.shape[1]))
            outputs1 = net(x)
            if(i==0): final_y=outputs1.cpu().detach().numpy()
            else:final_y=np.vstack([final_y,outputs1.cpu().detach().numpy()])
            #torch.cat((final_y,outputs), 0)
            
            loss1 = criterion(outputs1,label1)
            running_loss1 += loss1.item()
            
        final_y=torch.from_numpy(final_y.astype(np.float32)).to(device)
        max_vals,Truths= torch.max(Y_test[:,0:classnum],1)
        acc2=mc.calc_accuracy(final_y[:,0:classnum],Truths)
        
        loss_accbuf.append([running_loss,acc1,acc2,running_loss1])
        
        if epoch % 50==0:
            t2=time.time()
            print('time: %0.3f'% (t2-t1))
            print('[%d, %5d] Tloss: %0.6f,Vloss: %0.6f, training acc %0.6f, valditing acc %0.6f' % (epoch + 1, i + 1, running_loss,running_loss1,acc1,acc2))
        #if(running_loss<0.02):return
        running_loss = 0.0
        
    print('time = %2dm:%2ds' % ((time.time() - start)//60, (time.time()-start)%60))
    torch.save(net.state_dict(),'model_weights/weights_MLP_'+ms+'_%d' % epochs+'_'+str(alaph)+'.pth') 
    np.save('model_weights/Loss_data_MLP_'+ms+'_'+str(alaph)+'_' +str(epochs)+'.npy',np.array(loss_accbuf))

def plot_accuracy(lossaccbf,sfname='',ds=100):
    fig = plt.figure(figsize=(6,4))
    ax1 = fig.add_subplot(111)
    ax1.spines['top'].set_linewidth(0);##设置底部坐标轴的粗细
    ax1.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax1.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    ax1.spines['right'].set_linewidth(0);##设置左边坐标轴的粗细
    xs=np.arange(0,len(lossaccbf))
    plt.yticks(fontsize=10)#,fontweight='bold'
    plt.xticks(fontsize=10)
    plt.xlim(1,len(lossaccbf))
    plt.xscale('log')
    plt.ylim(0,1)
    st=0#int(len(lossaccbf)/20)
    sp=int(len(lossaccbf)/ds)
    ax1.plot(xs[st::sp]+1,lossaccbf[st::sp,0],'k--',marker='d',markersize=3,markerfacecolor='none',linewidth=1,label='Loss')
    ax1.plot(xs[st::sp]+1,lossaccbf[st::sp,1],'b--',marker='o',markersize=3,linewidth=1,markerfacecolor='none',label='Accuracy')
    #ax1.set_xlabel('epochs',mc.font1)
    #ax1.set_ylabel('Training loss',mc.font1)  # 可以使用中文，但需要导入一些库即字体
    #plt.title('ROC Curve for class '+ str(class_id))
    ax1.legend(loc='upper left',edgecolor='w')
    
    # ax2 = ax1.twinx() 
    # ax2.spines['top'].set_linewidth(2);##设置底部坐标轴的粗细
    # ax2.spines['bottom'].set_linewidth(2);##设置底部坐标轴的粗细
    # ax2.spines['left'].set_linewidth(2);##设置左边坐标轴的粗细
    # ax2.spines['right'].set_linewidth(2);##设置左边坐标轴的粗细
    # ax2.plot(xs[st::sp],lossaccbf[st::sp,1],'b--',marker='o',markersize=3,linewidth=1,markerfacecolor='none',label='Training accuracy')
    # #ax2.plot(xs[st::sp],lossaccbf[st::sp,2],'m--',marker='o',markersize=3,linewidth=1,markerfacecolor='none',label='Testing Accuracy')
    # plt.xticks(fontsize=10,fontweight='bold')
    # plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=10,fontweight='bold')
    # #plt.xlabel('epochs',mc.font1)
    # ax2.set_ylabel('Accuracy',mc.font1) # 可以使用中文，但需要导入一些库即字体
    # #plt.title('ROC Curve for class '+ str(c
    # ax2.legend(loc='center right',edgecolor='w',bbox_to_anchor=(1.02,0.54))

    print('loss:',lossaccbf[-1,0], 'Trianing acc:',lossaccbf[-1,1],'Valdit acc:',lossaccbf[-1,2])
    
    plt.savefig('saved_figs/'+sfname+'_lossacc.png',bbox_inches='tight', dpi=300) 


def plot_confusion_matrix(org_pred,orgLabel,sfname='',classnames='',figsize=[3,3],cmap=plt.cm.Blues):  
    max_vals,Truths= torch.max(orgLabel,1)
    Ps=org_pred.reshape(org_pred.shape[0],org_pred.shape[-1])
    max_vals,Preds= torch.max(Ps,1)
    classes={0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q',5: 'S', 6: 'N', 7: 'S'}
    
    cm = confusion_matrix(y_true=Truths.cpu().detach().numpy(), y_pred=Preds.cpu().detach().numpy())
    
    plt.figure(figsize=(figsize[0],figsize[1]))
    title='Confusion matrix'
    mc.plot_confusion_matrix(cm=cm,normalize=True,classes=classnames,cmap=cmap)
    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=0,fontsize=10)
    plt.yticks(tick_marks, classnames, rotation=0,fontsize=10)
    
    buf=[]
    for i in range(0,cm.shape[0]):
        buf.append(cm[i,i])
        
    acc=np.round(np.sum(buf)/np.sum(cm),3)
    #print('#############',cm)
    plt.title("Accuracy="+str(acc),mc.font1)
    # plt.colorbar()
    plt.savefig('saved_figs/'+sfname+'_confused_matrix.png',bbox_inches='tight', dpi=300) 
    
# Train_model()
lossaccbf=np.load('model_weights/Loss_data_MLP_'+ms+'_'+str(alaph)+'_' +str(epochs)+'.npy') 
net.load_state_dict(torch.load('model_weights/weights_MLP_'+ms+'_%d' % epochs+'_'+str(alaph)+'.pth'))     


for i in range(int(Y_test.shape[0]/Batchsize)+1):
    buf=X_test[Batchsize*i:Batchsize*(i+1),:]
    x=torch.reshape(buf,(buf.shape[0],buf.shape[1]))
    outputs = net(x)
    if(i==0): final_y=outputs.cpu().detach().numpy()
    else:final_y=np.vstack([final_y,outputs.cpu().detach().numpy()])

    
final_y=torch.from_numpy(final_y.astype(np.float32)).to(device)
max_vals,Truths= torch.max(Y_test[:,0:classnum],1)
overall_acc=mc.calc_accuracy(final_y[:,0:classnum],Truths)
print('Overall Accuracy',overall_acc)
orgLabel=Y_test[:,0:classnum]
Preds=final_y[:,0:classnum]


plot_accuracy(lossaccbf,sfname='ANN',ds=1000)

plot_confusion_matrix(Preds,orgLabel,sfname='ANN',classnames=mc.yls[:6],figsize=[2.8,2.8])