#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import linalg as LA
import random
import time
import pdb
import unittest
from PIL import Image
from pyspark import SparkContext, SparkConf
import time
import csv
import matplotlib.pyplot as plt



class LSH:

    def __init__(self, filename, k, L):
        """
        Initializes the LSH object
        filename - name of file containing dataframe to be searched
        k - number of thresholds in each function
        L - number of functions
        """
        conf = SparkConf()
        self.sc = SparkContext().getOrCreate(conf = conf)
        self.k = k
        self.L = L
        self.A = self.load_data(filename)
        self.functions = self.create_functions()
        self.hashed_A = self.hash_data()
        
    def __del__(self):
        self.sc.stop()
        
    def l1(self, u, v):
        """
        Finds the L1 distance between two vectors
        u and v are 1-dimensional Row objects
        """
        return LA.norm(np.array(u) - np.array(v), ord = 1)#for getting the l1 distance which is manhatten distance
        raise NotImplementedError

    def load_data(self, filename):
        """
        Loads the data into a spark DataFrame, where each row corresponds to
        an image patch -- this step is sort of slow.
        Each row in the data is an image, and there are 400 columns.
        """
        lines = self.sc.textFile(filename).map(lambda v: v.split(','))# splits the data by "," and stores in rdd
        l = lines.map(lambda w: [int(float(c)) for c in w]).zipWithIndex()# just adding the index for each row 
        return l
        raise NotImplementedError

    def create_function(self, dimensions, thresholds):
        """
        Creates a hash function from a list of dimensions and thresholds.
        """
        def f(v):
            s = ''
            for i in range(len(dimensions)):
                if (v[dimensions[i]] >= thresholds[i]):# returns a string value that is described in the create_functions() 
                    s += '1'
                else:
                    s += '0'           
            return s
            raise NotImplementedError
        return f

    def create_functions(self, num_dimensions=400, min_threshold=0, max_threshold=255):
        """
        Creates the LSH functions (functions that compute L K-bit hash keys).
        Each function selects k dimensions (i.e. column indices of the image matrix)
        at random, and then chooses a random threshold for each dimension, between 0 and
        255.  For any image, if its value on a given dimension is greater than or equal to
        the randomly chosen threshold, we set that bit to 1.  Each hash function returns
        a length-k bit string of the form "0101010001101001...", and the L hash functions 
        will produce L such bit strings for each image.
        """
        functions = []
        for i in range(self.L):
            dimensions = np.random.randint(low = 0, 
                                    high = num_dimensions,
                                    size = self.k)
            thresholds = np.random.randint(low = min_threshold, 
                                    high = max_threshold + 1, 
                                    size = self.k)

            functions.append(self.create_function(dimensions, thresholds))
        return functions

    def hash_vector(v):
        
        """
        Hashes an individual vector (i.e. image).  This produces an array with L
        entries, where each entry is a string of k bits.
        """
        # never used this function
        raise NotImplementedError

    def hash_data(self):
        """
        Hashes the data in A, where each row is a datapoint, using the L
        functions in 'self.functions'
        """
        fun = self.functions #gets the functions into fun
        qwerty = self.A.map(lambda q: q + ([f(q[0]) for f in fun],))# just adding the hash functions to the rdd file
        return qwerty
        # you will need to use self.A for this method
        raise NotImplementedError

    def get_candidates(self, query_index):
        """
        Retrieve all of the points that hash to one of the same buckets 
        as the query point.  Do not do any random sampling (unlike what the first
        part of this problem prescribes).
        Don't retrieve a point if it is the same point as the query point.
        """
        # set() & set() gives the difference between the query_index hash function
        # and any function give me true if there an element in it and flase if it has empty string
        # fetches the data if it is true and stores it into a bucket rdd
        bucket = self.hashed_A.filter(lambda x: (x[2] != query_index[2]) and (any(set(x[2]) & set(query_index[2]))))
        return bucket
        
        # you will need to use self.hashed_A for this method
        raise NotImplementedError

    def lsh_search(self, query_index, num_neighbors):
        """
        Run the entire LSH algorithm
        """
        def l1(u, v):
            return LA.norm(np.array(u) - np.array(v), ord = 1)
    
        start = time.time()# for getting the time 
        buck = self.get_candidates(query_index)# will call the get_candidates and stores into buck
        dist = buck.map(lambda x: x + (l1(x[0], query_index[0]),))# adding the distance between the quer_index and candidates
        sor = dist.map(lambda x: (x[3],x[1]))# it is a tuple of distance and index
        w = sor.sortByKey()# sorts the data according to the distance
        ti = time.time() - start
        return (w.take(num_neighbors),ti)
        
        raise NotImplementedError


#function to plot images
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")


# implementation of linear_search where it compares with all other vectors  
def linear_search(A, query_index, num_neighbors):
    neigh = {}
    
    def l1(u,v):
        return LA.norm(u-v, ord=1)
    start = time.time()
    for i in range(len(A)):
        dis = l1(A[i],query_index)
        if (dis == 0):
            continue
        else:
            neigh[i] = dis
    
    ti = time.time() - start
    return (sorted(neigh.items(), key = lambda kv:(kv[1],kv[0]))[:num_neighbors] ,ti)
    raise NotImplementedError



# gives the error
def lsh_error(lsh_distance, linear_dis):
    total = 0.0
    for i in range(10):
        total += (lsh_distance[i]/linear_dis[i])
    return total/10
    raise NotImplementedError


if __name__ == '__main__':
    # create an LSH object using lsh = LSH(k=16, L=10)
    """
    Your code here
    """
    with open('patches.csv','r') as f:
        A = list(csv.reader(f, delimiter= ","))
    A = np.array(A[:],dtype = np.float)

    lsh = LSH('patches.csv', k=24, L=10)# calling the lsh class
    
    # for ploting the images of both the methods 
    f,t = lsh.lsh_search(lsh.hashed_A.collect()[99], 10)
    row_lsh = []
    for i in range(len(f)):
        row_lsh.append(f[i][1])
    plot(A,row_lsh,'img_lsh')
    # for linear image plotting    
    row_linear = []
    ld,lt = linear_search(A, A[99], 10)
    for i in range(len(f)):
        row_linear.append(ld[i][0])
    plot(A,row_linear,'img_linear')
        
    # looping through L and getting the change in errors
    error_l = []
    for L in range(10,22,2):
        lsh_distance = []
        linear_dis = []
        lsh_time = []
        linear_time = []
        lsh = LSH('patches.csv', k=24, L=L)
        for i in range(99,1000,100):
            distance = []
            f,t = lsh.lsh_search(lsh.hashed_A.collect()[i], 3)
            for i in range(len(f)):
                distance.append(f[i][0])
            lsh_distance.append(sum(distance))
            lsh_time.append(t)
        
            lid = []
            ld,lt = linear_search(A, A[i], 3)
            for i in range(len(ld)):
                lid.append(ld[i][1])
            linear_dis.append(sum(lid))
            linear_time.append(lt)
            
        lsh.sc.stop()# stoping the sparkcontext
        error_l.append(lsh_error(lsh_distance, linear_dis))
    
    # looping through k and getting the change in errors 
    error_k = []
    for k in range(16,26,2):
        lsh_distance = []
        linear_dis = []
        lsh_time = []
        linear_time = []
        lsh = LSH('patches.csv', k=k , L=10)
        for i in range(99,1000,100):
            distance = []
            f,t = lsh.lsh_search(lsh.hashed_A.collect()[i], 3)
            for i in range(len(f)):
                distance.append(f[i][0])
            lsh_distance.append(sum(distance))
            lsh_time.append(t)
        
            lid = []
            ld,lt = linear_search(A, A[i], 3)
            for i in range(len(ld)):
                lid.append(ld[i][1])
            linear_dis.append(sum(lid))
            linear_time.append(lt)
            
        lsh.sc.stop()   
        error_k.append(lsh_error(lsh_distance, linear_dis))
        

    '''    
    print("Average Time for LSH search: ", sum(lsh_time)/10)
    print("Average Time for Linear search: ", sum(linear_time)/10)
        
    print("Error is :",lsh_error(lsh_distance, linear_dis))
    '''

    #plotting the L vs Error graph
    plt.plot([10,12,14,16,18,20], error_l)
    plt.xlabel('function of L')
    plt.ylabel('Error')
    plt.title('L vs Error')
    plt.show()


    #plotting the k vs Error graph
    plt.plot([16,18,20,22,24], error_k)
    plt.xlabel('function of K')
    plt.ylabel('Error')
    plt.title('K vs Error')
    plt.show()



    plot(A,[99,100],'img')# plotting the 100th label





