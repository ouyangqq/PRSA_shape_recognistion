# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:10:01 2022

@author: qiang
"""

## importing the required packages
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection)
## Loading and curating the data
import metrics as mc


tlabels=mc.yls


A=np.load("gdata/shapes_training.npy")
[Train,validt]=A.item(),A.item()
#[Train,validt]=[np.load('gdata/Data_train_ML.npy'),0]#Train: 训练数据集, Test: 测试数据集
X =Train["data"].reshape(Train["data"].shape[0],Train["data"].shape[1]*Train["data"].shape[2])
#X=np.load('gdata/Shapes_PCA.npy')

Lb =Train["label"]
y=np.zeros(len(Lb[:,0]))
for i in range(Lb.shape[1]):y=y+i*Lb[:,i]



'''
A=np.load("../CNN_Majiang/gdata/Majiang_training.npy")
[Train,validt]=A.item(),A.item()
#[Train,validt]=[np.load('gdata/Data_train_ML.npy'),0]#Train: 训练数据集, Test: 测试数据集
X =Train["data"].reshape(Train["data"].shape[0],Train["data"].shape[1]*Train["data"].shape[2])
Lb =Train["label"]
y=np.zeros(len(Lb[:,0]))
for i in range(Lb.shape[1]):y=y+i*Lb[:,i]
'''


n_samples, n_features = X.shape
n_neighbors = 30
## Function to Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)    


    plt.figure()
    ax = plt.subplot(111)
    '''
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]), color=plt.cm.Set1(y[i] / 10.),   fontdict={'weight': 'bold', 'size': 9})
    '''
    if hasattr(offsetbox, 'AnnotationBbox'):
       ## only print thumbnails with matplotlib> 1.0
       shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
        ## don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(Train["data"][i], cmap=plt.cm.gray_r),X[i])
    ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    
    if title is not None:plt.title(title)


'''
#----------------------------------------------------------------------
## Plot images of the digits
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
   ix = 10 * i + 1
   for j in range(n_img_per_row):
      iy = 10 * j + 1
      img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')
'''
'''
## Computing PCA
print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,
              "Principal Components projection of the digits (time %.2fs)" %
              (time() - t0))
'''

# Computing t-SNE

# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=3)
# t0 = time()
# X_tsne = tsne.fit_transform(X)
# np.save("gdata/shapes_tsne.npy",X_tsne)
# print('time consume:',time() - t0)



X_tsne=np.load("gdata/shapes_tsne.npy")
X_tsne=(X_tsne-X_tsne.min())/(X_tsne.max()-X_tsne.min())

np.save('gdata/shape_for_trianing_ANN.npy',{0:X_tsne,1:Lb})

#plot_embedding(X_tsne,"t-SNE embedding of the digits (time %.2fs)" %(time() - t0))

ax = plt.figure(figsize=(2.85,2.85))

ax=plt.subplot(1,1,1)  
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细

lgs=np.arange(len(set(y)))#[[0,3,5]]


for m,c,i,target_name in zip(mc.markers,mc.colors,lgs,tlabels):
    print(target_name)
    plt.scatter(X_tsne[y==i,0], X_tsne[y==i, 1],c='w',edgecolors=c,marker='o',label=target_name,s=5)

plt.xticks([0,0.25,0.50,0.75,1.0],fontsize=10,rotation=0)#,fontweight='bold'
plt.yticks([0,0.25,0.50,0.75,1.0],fontsize=10,rotation=0)
plt.xlim(0,1)
plt.ylim(0,1)

#plt.xlabel('PC1',fontsize=10,fontweight='bold')
#plt.ylabel('PC2',fontsize=10,fontweight='bold')
plt.legend(bbox_to_anchor=(1.24, 0.99),loc=1,ncol=1,fontsize=10)
#plt.legend(bbox_to_anchor=(0.35, 0.99),ncol=1)loc='lower right'
 

#plt.title('Cluster')

plt.savefig('saved_figs/tsne_'+str(len(lgs))+'_kind.png',bbox_inches='tight', dpi=300)
plt.show()



#plt.imshow(X[0,:].reshape(8,8))