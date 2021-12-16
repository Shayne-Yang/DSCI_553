import math

import numpy as np
from random import sample
from sklearn.cluster import KMeans
from math import floor
from math import sqrt
from itertools import combinations
import sys

class BFR_clustering(object):
    def __init__(self,array,cluster_num):
        """
        :param array: a numpy array
        :param cluster_num: the number of cluster you want
        """
        self.array = array
        self.cluster_num = cluster_num #the default should be 10
        self.index = list(i for i in range(array.shape[0])) # a list of index
        self.X = array[:,2:] # the matrix of X
        self.dimension = self.X.shape[1]

    def random_sampling(self,index,sample_size):
        sample_index = sample(index,sample_size) # this will output a list
        sample_X = self.array[sample_index,:]
        return sample_index,sample_X # return the sample index and sample X for each time

    def rough_k_means(self,sample_index):
        """
        :param sample_index: sample index based on the population
        :return: get a dictionary, gives the cluster number and its corresponding sample_index list
        """
        cluster_result = {}
        if len(sample_index) <= self.cluster_num * 5:
            cluster_result = {i:[index] for i,index in enumerate(sample_index)}
        else:
            sample = self.X[sample_index,:]
            kmeans = KMeans(n_clusters=self.cluster_num * 5,random_state=0).fit(sample)
            labels = kmeans.labels_
            cluster_result = {key:[] for key in np.unique(labels)}
            for i in range(len(labels)):
                cluster_result[labels[i]].append(sample_index[i])
        return cluster_result

    def precise_k_means(self,sample_index):
        """
        the precise cluster, use the exact same number of clusters we want
        :param sample_index:the index you want to cluster
        :return: a dict. {cluster_label: [point_index]}
        """
        sample = self.X[sample_index, :]
        kmeans = KMeans(n_clusters=self.cluster_num,random_state=0).fit(sample)
        labels = kmeans.labels_
        cluster_result = {key:[] for key in np.unique(labels)}
        for i in range(len(labels)):
            cluster_result[labels[i]].append(sample_index[i])
        return cluster_result

    def single_cluster(self,cluster_dic):
        """

        :param cluster_dic:the dict we get from the k_means function
        :return: 1 list: the index which have no other index in the same cluster
        1 cluster: the cluster dict without cluster with only one point
        """
        single_index = []
        single_cluster = []
        for key in cluster_dic.keys():
            if len(cluster_dic[key]) == 1:
                single_index += cluster_dic[key]
                single_cluster.append(key)
        cluster_dic = {key:cluster_dic[key] for key in cluster_dic.keys() if key not in single_cluster}
        # get the cluster dictionary where have more than 1 points
        return single_index,cluster_dic

    def DS_cluster(self,cluster_dic): # discard the point and gain its information
        """

        :param cluster_dic:the cluster dict without cluster with only one point
        :return: a dict: {cluster_id:[num_of_point, np.sum_vector,np.sum_vector]}
        """
        ds_dic = {}
        for key,value in cluster_dic.items():
            num_point = len(value)
            matrix = self.X[value,:]
            sum_vector = matrix.sum(axis=0) # 1-d numpy array
            sumsq_vector = (matrix*matrix).sum(axis=0) # 1-d numpy array
            ds_dic[key] = [num_point,sum_vector,sumsq_vector]
        return ds_dic

    def Mahalanobis_distance(self,point,cluster):
        """
        :param point:the point you want to assgin 1-d np-array
        :param cluster: the cluster information.a tuple or a list inclue three elements:
        [num,SUM,SUMSQ]
        :return: the Mahalanobis_distance
        """
        centrual_vector = cluster[1]/cluster[0]
        std_vector = np.sqrt((cluster[2]/cluster[0]) - (cluster[1]/cluster[0])**2)
        return sqrt(sum(((point - centrual_vector)/std_vector) ** 2))

    def check_point(self,point_index,DS_dict):
        """
        check if the point could be assign to some of those sets
        :param point_index:the index of the point
        :param DS_dict: the ds summary
        :return: the cluster index, if there's no such cluster, return -1
        """
        point_vector = self.X[point_index,:]
        smallest_distance = 10000 #genearte a very large number
        best_cluster = -1 # generate the initial cluster index
        if DS_dict:
            for key,vector in DS_dict.items():
                try:
                    self.Mahalanobis_distance(point_vector, vector)
                except IndexError as ie:
                    print(point_vector)
                    print(vector)
                M_distance = self.Mahalanobis_distance(point_vector,vector)
                if M_distance < smallest_distance and M_distance < 2*math.sqrt(self.dimension):
                    smallest_distance = M_distance
                    best_cluster = key
        return best_cluster

    def assign_point(self,point_list,DS_dict,CS_dict):
        """

        :param point_list: index of point need to assign
        :param DS_dict: current ds summary
        :param CS_dict: current cs summary
        :return: two dict, 1 list
        """
        point_ds = {key:[] for key in DS_dict.keys()}#{cluster_index:[point_index]}
        cs_list =[] #point cannot assgin to point_ds
        point_cs = ({key:[] for key in CS_dict.keys()} if CS_dict else {})#{cluster_index:[point_index]}
        point_rs = []#[point_index]
        for point in point_list:
            cluster = self.check_point(point,DS_dict)
            if cluster != -1:
                point_ds[cluster].append(point)
            else:
                cs_list.append(point)
        if cs_list and CS_dict:
            for point in cs_list:
                cluster = self.check_point(point,CS_dict)
                if cluster != -1:
                    point_cs[cluster].append(point)
                else:
                    point_rs.append(point)
        elif cs_list:
            point_rs += cs_list

        return point_ds,point_cs,point_rs

    def renew_dict(self,point_dict,DS_dict,ds_cluster):
        if point_dict:
            for key,points in point_dict.items():
                summary_vector = [len(points),sum(self.X[index,:] for index in points)
                                  ,sum(self.X[index,:]*self.X[index,:] for index in points)]
                DS_dict[key] = [DS_dict[key][k]+summary_vector[k] for k in range(3)]
                ds_cluster[key] += points
        return DS_dict, ds_cluster

    def Mahalanobis_distance_cluster(self,cluster1,cluster2):
        """

        :param cluster1:the summary of cluster1:[num,SUM,SUMSQ]
        :param cluster2:the sumary of cluster2:[num,SUM,SUMSQ]
        :return: the mahalanobis distance between these two clusters
        """
        point = cluster1[1]/cluster1[0]
        return self.Mahalanobis_distance(point,cluster2)


    def merge_cs_cluster(self,cs_cluster1,cluster1_detail,cs_cluster2,cluster2_detail):
        """

        :param cs_cluster1: the cluster you gain from the previous-- dict summary
        :param cluster1_detail: the index of points for each cluster
        :param cs_cluster2: the cluster you gain from the RS-- dict summary
        :param cluster2_detail: the index of points for each cluster
        :return: one cs_cluster dict where there's no two cluster that has the M_distance smaller than 2*sqrt(d)
        """
        if len(cs_cluster1) != 0 and len(cs_cluster2) != 0:
            cs_cluster2_adj = {key+max(cs_cluster1.keys()):cs_cluster2[key] for key in cs_cluster2}
            raw_cluster = {**cs_cluster1,**cs_cluster2_adj} # merge two dict into one
            cluster2_detail_adj = {key+max(cluster1_detail.keys()):cluster2_detail[key] for key in cs_cluster2}
            raw_cluster_detail = {**cluster1_detail,**cluster2_detail_adj}
            raw_index = list(raw_cluster.keys())
            d = self.dimension
            while True:
                min_dist = math.inf
                drop_set = tuple()
                for (set1,set2) in combinations(raw_cluster.keys(),2):
                    dist = self.Mahalanobis_distance_cluster(raw_cluster[set1],raw_cluster[set2])
                    if dist <=min_dist:
                        min_dist = dist
                        drop_set = (set1,set2)
                if min_dist > 2*math.sqrt(d):
                    break
                else:
                    set1,set2 = drop_set
                    try:
                        raw_cluster[set1][0] += raw_cluster[set2][0]
                    except KeyError:
                        print(set1,set2,raw_cluster)
                    raw_cluster[set1][1] += raw_cluster[set2][1]
                    raw_cluster[set1][2] += raw_cluster[set2][2]
                    raw_cluster.pop(set2) # drop the second set but remain the first one
                    raw_cluster_detail[set1] += raw_cluster_detail[set2]
                    raw_cluster_detail.pop(set2)
            return raw_cluster,raw_cluster_detail
        elif len(cs_cluster2) != 0:
            return cs_cluster2,cluster2_detail
        elif len(cs_cluster1) != 0:
            return cs_cluster1,cluster1_detail
        else:
            return {},{}


    def fit(self):
        """
        :return: inital DS,CS statistical summary, RS point, remain_point
        """
        discard_point = []  # get the index of point we discard
        cs_point = []  # get the index of point in the compression set
        rs_point = []  # get the index of point in the retained set
        remain_point = self.index  # get the index of point we do not tackle currently
        verbose = [] # intermediate output
        total_size = self.array.shape[0]
        sample_size = floor(total_size/5)
        initial_index, _ = self.random_sampling(self.index,sample_size)
        step2_output = self.rough_k_means(initial_index)
        step3_point, _ = self.single_cluster(step2_output) # get the rs point in the initial matrix
        # initial discard point set
        restofpoint = [index for index in initial_index if index not in step3_point]
        # step 4
        initial_cluster = self.precise_k_means(restofpoint)
        # step 5
        ds_dic = self.DS_cluster(initial_cluster) # get statistic information about discard set
        ds_cluster = initial_cluster
        # step 6
        step3_point_cluster = self.rough_k_means(step3_point)
        rs, cs_cluster = self.single_cluster(step3_point_cluster)
        cs_dic = self.DS_cluster(cs_cluster) # get statistic information about the compression set
        # from now on, the initial step has completed

        # summuraize the data: “the number of the discard points”,
        # “the number of the clusters in the compression set”,
        # “the number of the compression points”
        # “the number of the points in the retained set”
        verbose.append(['Round 1:', len(restofpoint),(len(cs_dic.keys) if cs_dic else 0),len(step3_point) - len(rs),len(rs)])
        remain_point = list(set(remain_point) - set(initial_index))

    #     return ds_dic, cs_dic, rs,remain_point
    #
    # def merge_point(self,ds_dic,cs_dic,rs,remain_point,verbose):
    #     """
    #     :param ds_dic: the ds summary from the initialization
    #     :param cs_dic: the cs summary from the initialization
    #     :param rs: the rs data index
    #     :param remain_point: the index of remain points
    #     :return:
    #     """
        i = 4
        while len(remain_point) > 0 :
            sample_size = len(remain_point) // i
            sample_index,_ = self.random_sampling(remain_point,sample_size)
            remain_point = list(set(remain_point) - set(sample_index))
            i = i - 1
            point_ds, point_cs, point_rs = self.assign_point(sample_index,ds_dic,cs_dic)
            ds_dic,ds_cluster = self.renew_dict(point_ds,ds_dic,ds_cluster)
            cs_dic,cs_cluster = self.renew_dict(point_cs,cs_dic,cs_cluster)
            rs += point_rs
            new_cs = self.rough_k_means(rs) # generate new cs clusters from the rs
            rs,new_cs_cluster = self.single_cluster(new_cs)
            new_cs_dic = self.DS_cluster(new_cs_cluster)#get the renew rs, and new cs dict
            # step 12
            cs_dic,cs_cluster = self.merge_cs_cluster(cs_dic,cs_cluster,new_cs_dic,new_cs_cluster)
            verbose.append([f'round {5-i}:',sum(value[0] for value in ds_dic.values()),(len(cs_dic.keys()) if cs_dic else 0),
                            sum(value[0] for value in cs_dic.values()),
                            len(rs)])
    # final step
        _,point_ds = self.merge_cs_cluster(ds_dic,ds_cluster,cs_dic,cs_cluster)
        output_result = list()
        for cluster_id,index_list in point_ds.items():
            if cluster_id > self.cluster_num: # those are retained number
                for num in index_list:
                    output_result.append((num,-1))
            else:
                for num in index_list:
                    output_result.append((num,cluster_id))

        return verbose,sorted(output_result,key=lambda x: x[0])

if __name__ == "__main__":


    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    # input_file = 'hw6_clustering.txt'
    # n_cluster = 5
    # output_file = 'test.txt'

    data = np.genfromtxt(input_file,delimiter=',')
    data[:,0] = data[:,0].astype(np.int32)
    BFR = BFR_clustering(data,n_cluster)
    verbose, output = BFR.fit()


    with open(output_file,'w+') as f:
        f.write("The intermediate results:"+"\n")
        for line in verbose:
            f.write(line[0] + ','.join(str(line[i]) for i in range(1,5)) + '\n')
        f.write("\nThe clustering results:\n")
        for item in output:
            f.write(','.join([str(i) for i in item]) + "\n")
    f.close()





















