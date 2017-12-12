# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:29:04 2017

@author: oeasy
"""

import os, sys
import numpy
import time
import matplotlib.pyplot as plt
import math


def euclidean_distance(vec1, vec2):
    '''
    calculate euclidean distance
    '''
    return math.sqrt(sum(math.power(vec1-vec2, 2)))


def init_centroids(dataset, num_clusters):
    num_samples, dims = dataset.shape
    
    print(num_clusters, dims)
    centroids = zeros(num_clusters, dims)
    
    for i in range(k):
        index = int(random.uniform(0, num_samples))
        centroids[i,:] = dataset[index, :]
    
    return centroids


def kmeans(dataset, num_clusters):
    num_samples = dataset.shape[0]
    
    cluster_assment = mat(zeros((num_samples, 2)))
    cluster_changed = True
    
    centroids = init_centroids(dataset, num_clusters)
    
    while cluster_changed:
        cluster_changed = False
        
        for index_sample in range(num_samples):
            min_distance = 10000.0
            min_index = 0
            
            for index_cluster in range(num_clusters):
                distance = euclidean_distance(
                    centroids[index_cluster, :], 
                    dataset[index_sample,:]
                )
                if distance < min_distance:
                    min_distance = distance
                    min_index = index_cluster
            if cluster_assment[index_sample, :] != min_index:
                cluster_changed = True
                cluster_assment[index_sample, :] = min_index, min_distance ** 2
        for index_cluster in range(num_clusters):
            points_in_cluster = dataset[nonzero(cluster_assment[:, 0].A == index_cluster)[0]]
            centroids[index_cluster, :] = mean(points_in_cluster, axis=0)
    
    print('cluster complete')
    return centroids, cluster_assment


def show_clusters(
        dataset, 
        num_cluster, 
        centroids, 
        cluster_assment):
    num_samples, dims = dataset.shape
    if dims != 2:
        print('dataset dimension is not 2D')
        return 1
    
    mark = ['or', 'ob' 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for index_sample in xrange(num_samples):
        mark_index = int(cluster_assment[index_sample, 0])
        plt.plot(
            dataset[index_sample, 0], 
            dataset[index_sample, 1], 
            mark[mark_index%len(mark)]
        )
    
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):  
        plt.plot(
            centroids[i, 0], 
            centroids[i, 1], 
            mark[i], 
            markersize = 12
        )
    
    plt.show()


if __name__ == "__main__":
    dataSet = []  
    fileIn = open('E:/test_ml/test_dataset.txt')  
    for line in fileIn.readlines():  
        lineArr = line.strip().split(',')  
        dataSet.append([float(lineArr[0]), float(lineArr[1])])  
      
    ## step 2: clustering...  
    print("step 2: clustering...")
    dataSet = mat(dataSet)  
    k = 4  
    centroids, clusterAssment = kmeans(dataSet, k)  
      
    ## step 3: show the result  
    print("step 3: show the result...")
    show_clusters(dataSet, k, centroids, clusterAssment)  
