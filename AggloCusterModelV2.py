import pandas as pd
import numpy as np
from itertools import combinations

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))
    

class AgloClusterModel:
    def __init__(self, data, threshold, link='single'):
        self.data = data
        self.threshold = threshold
        self.distances = [np.apply_along_axis(distanceFrom(point), 1, data) for point in data]
        self.link = link
        self.tree = {}
        self.clusters = set()
        
    def findClosestPair(self):
        # returns index of 2-d array where the min value is 
        # might be minus 1 
        clusters = list(self.clusters)
        min_ = np.where(self.distances == np.min(self.distances))
        return( min_[0], min_[1])

    def measuring(self): 
        # split tree based on threshold 
        pass 

    def distance_between_clusters(self, c1, c2): 
        min_ = distance(c1[0], c2[0])
        for i in c1: 
            for j in c2: 
                d = distance(i, j)
                if d < min_: 
                    min_ = d 
        return min_

    def go_through_trees(self, cluster): 
        for tree in self.tree: 
            return self.find_item_in_tree(tree)

        # searches list of trees and all trees for cluster 
            # if found 
                # removes the indv leaf or node from the tree list 
                # returns the node or leaf 
                
            # else 
                # returns a new leaf 
                # i.e { type:"leaf", "height":0, "data": "{}".format(self.data[c1])}


    def find_item_in_tree(self, item, tree): 
        pass
            

    def build(self): 
        i = 0
        for i in range(len(self.data)): 
            self.clusters.append([i])
            i+= 1 
        # self.clusters [[1], [2,3,4], [5]] -> data[1],data[2],data[3] ect are in the same cluster


        while len(self.tree) > 1: 
            # all the unique combinations of clusters [([1], [2, 3, 4]), ([1], [5]), ([2, 3, 4], [5])]
            cluster_combinations = [comb for comb in combinations(self.clusters, 2)]
            # all of the distances for cluster combinations [1.2, 1.3, 1.8]
            cluster_distances = []
            for combo in cluster_combinations: 
                cluster_distances.append(self.distance_between_clusters(combo[0], combo[2]))
            
            min_distance = min(cluster_distances)
            clusters = cluster_combinations.index(min(cluster_distances)) # ([1], [2, 3, 4])
            c1 = clusters[0] #[1]
            c2 = clusters[1] #[2, 3, 4]

            if len(c1) == 1 and len(c2) == 1: 
                leaf1 = { type:"leaf", "height":0, "data": "{}".format(self.data[c1])}
                leaf2 = { type:"leaf", "height":0, "data": "{}".format(self.data[c2])}
                node = {type: "node", "height": min_distance, "nodes": [leaf1, leaf2]}
                joined_clusters = c1 + c2
                self.clusters.remove(c1)
                self.clusters.remove(c2)
                self.clusters.append(joined_clusters)
                self.tree.append(node)

            else: 
                node1= self.search_tree_for_node(c1)
                node2= self.search_tree_for_node(c2)
                node = {type: "node", "height": min_distance, "nodes": [node1, node2]}
                self.tree.append(node)

        
        self.tree = self.tree[0]


        


 

        
        
    



            
             



        
 

 