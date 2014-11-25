# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:34:48 2014

@author: colorbox
"""
from __future__ import division
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import random

irisdata=np.genfromtxt(r'C:\Users\colorbox\Downloads\iris.data',dtype='string',delimiter=',')
net=buildNetwork(4,20,1)
irises={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
ds=SupervisedDataSet(4,1)
targets=[]
for row in irisdata:
    random_val=random.random()
    iris_val=irises[row[4]]
    if random_val<.75:
        ds.addSample(tuple(row[0:4].astype('float')),(iris_val))
    else:
        row[4]=iris_val
        row=row.astype('float')
        targets.append(row)
trainer=BackpropTrainer(net,ds)
for i in range(100):
    print trainer.train()

right_count=0
total=0
for test_row in targets:
    guess=np.round(net.activate(test_row[0:4]))
    actual=test_row[4]
    if int(guess)==int(actual):
        right_count+=1
    total+=1
print right_count/total*100