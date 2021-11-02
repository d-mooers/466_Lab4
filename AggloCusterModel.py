import pandas as pd
import numpy as np
from itertools import combinations
import sys 
import json 


distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))
    

class AgloClusterModel:
    def __init__(self, data, threshold, link='single'):
        self.data = data
        print("data", self.data, type(self.data))
        self.threshold = threshold
        self.distances = [np.apply_along_axis(distanceFrom(point), 1, data) for point in data]
        self.link = link
        self.tree = []
        self.clusters = []
        
    def findClosestPair(self):
        # returns index of 2-d array where the min value is 
        # might be minus 1 
        min_ = np.where(self.distances == np.min(self.distances))
        return( min_[0], min_[1])

    def measuring(self, threshold): 
        # split tree based on threshold 
        # All nodes of the dendrogram with labels greater than α are removed
        # from the dendrogram, together with any adjacent edges.The resulting forest represents the clusters found by a hierarchical clustering
        # method that constructed the dendrogram, at the threshold α.
        
        # DFS 
        # ex: 7 
        # anything less than 7 in 7's node 
        # create new clusters [] based on the left and right 

        # Big idea = split into further clusters 

        # TESTING 
        # higher the the threshold, less clusters (ex: threshold = 100, same tree as before)

        # lower the threshold, way more clusters (ex: threshold = 3, WAY more trees)
        pass 

    def distance_between_clusters(self, c1, c2): 
        min_ = distance(self.data[c1[0]], self.data[c2[0]])
        for i in c1: 
            for j in c2: 
                d = distance(self.data[i], self.data[j])
                if d < min_: 
                    min_ = d 

        return min_

        # searches list of trees and all trees for cluster 
            # if found 
                # removes the indv leaf or node from the tree list 
                # returns the node or leaf 
                
            # else 
                # returns a new leaf 
                # i.e { type:"leaf", "height":0, "data": "{}".format(self.data[c1])}


    def go_through_trees(self, item): 
        # returns tree if item is found 
        
        i = 0
        for t in self.tree: 
            if self.find_item_in_tree(item, t): 
                # item is found in tree 
                tree = t.copy()
                # print("actual tree", self.tree[i])
                # print("copy tree", tree)
                
                # print("len " ,len(self.tree))
                del self.tree[i]
                # print("len " ,len(self.tree))
                return tree
            i+=1 

        # item is not found in any trees 
        # creates a leaf node 
        # return  { type:"leaf", "height":0, "data": "{}".format(self.data[item])}
            
            
    def find_item_in_tree(self, item, tree): 
        # item = [2,0] data point 
        # tree = dict 
        # checks if item is in tree and returns True or False 
  
        if "data" in tree: 
            
            tree_data = tree["data"]
            print("item", item)
            
            str_item  = json.dumps(item)
            
            if tree_data == str_item: 
            #BEFORE
            #if np.array_equal(list(tree["data"]),list(item)): 
                return True 

        else: 
            allNodes = tree['nodes']
            for node in allNodes: 

                if self.find_item_in_tree(item, node) == True: 

                    return True 

        return False 


    def create_tree_node(self, c):  # c = cluster [1] or [2,3,4] 
        if len(c)  == 1:  # [1] 
            #print("list(self.data[c[0]])", list(self.data[c[0]]), type(list(self.data[c[0]])))
            
            list_ = self.data[c[0]].tolist()
            print("list_", list_)
            json_string = json.dumps(list_)
            return { "type":"leaf", "height": "0", "data": json_string}
            
            # before 
            #return { "type":"leaf", "height": "0", "data": self.data[c[0]]}
            
        else: # [2,3,4] => pass in 2 
            return self.go_through_trees(list(self.data[c[0]]))
                

    def build(self): 
        i = 0
        for i in range(len(self.data)): 
            self.clusters.append([i])
            i+= 1 
        # self.clusters [[1], [2,3,4], [5]] -> data[1],data[2],data[3] ect are in the same cluster

        # print("initial clusters", self.clusters)
        while len(self.clusters) > 1: 
            # print("\nTREE", self.tree)
            print("while clusters", self.clusters)
            # all the unique combinations of clusters [([1], [2, 3, 4]), ([1], [5]), ([2, 3, 4], [5])]
            cluster_combinations = [comb for comb in combinations(self.clusters, 2)]
            # print("cluster combinations", cluster_combinations)
            # all of the distances for cluster combinations [1.2, 1.3, 1.8]
            cluster_distances = []
            
            for combo in cluster_combinations: 
                cluster_distances.append(self.distance_between_clusters(combo[0], combo[1]))

            # #print("cluster distances ", cluster_distances)
            
            min_distance = int(min(cluster_distances))
            #print("min distance ", min_distance)
            clusters = cluster_combinations[cluster_distances.index(min(cluster_distances))] # ([1], [2, 3, 4])
            #print("clusters of interest", clusters)
            
            c1 = clusters[0] #[1], pass in the actual data point 
            #print("c1", c1)
            c2 = clusters[1] #[2, 3, 4]
            #print("c2", c2)
            node1 = self.create_tree_node(c1)
            #print("\nnode1", node1)
            node2 = self.create_tree_node(c2)
            #print("\nnode2", node2)
            #node_list = [node1, node2]
            #json_string = json.dumps(node_list)
            #tree = {"type": "node", "height": "{}".format(min_distance), "nodes": json_string}
            tree = {"type": "node", "height": min_distance, "nodes": [node1, node2]}
            #print("\ntree 2", tree)
            #json_object2 = json.dumps(tree, indent = 4) 
            #json_object3 = json.dumps(node1, indent = 4) 
            #json_object4 = json.dumps(node2, indent = 4) 
            joined_clusters = c1 + c2
            

            self.clusters.remove(c1) # removes [1]
            self.clusters.remove(c2) # removes [2]
            self.clusters.append(joined_clusters) # adds [1,2]
            self.tree.append(tree)



            # if len(c1) == 1 and len(c2) == 1: 
            #     leaf1 = { type:"leaf", "height":0, "data": "{}".format(self.data[c1])}
            #     leaf2 = { type:"leaf", "height":0, "data": "{}".format(self.data[c2])}
            #     node = {type: "node", "height": min_distance, "nodes": [leaf1, leaf2]}
            #     joined_clusters = c1 + c2 # [1,2]
            #     self.clusters.remove(c1) # removes [1]
            #     self.clusters.remove(c2) # removes [2]
            #     self.clusters.append(joined_clusters) # adds [1,2]
            #     self.tree.append(node) 

            # else: 
            #     if len(c1) > 1: #[2,3]
            #     node1= self.search_tree_for_node(c1) # [1]
            #     node2= self.search_tree_for_node(c2) # [2,3]
            #     node = {type: "node", "height": min_distance, "nodes": [node1, node2]}
            #     self.tree.append(node)


        self.tree = self.tree[0]
        #print("\n clusters", self.clusters)

        print("\nend clusters", self.clusters)
        print("\ntree", self.tree, type(self.tree))
        
        #json_object = json.dumps(self.tree, indent = 4) 
        #print(json_object)


    def convert_tree_to_json(self, item, tree): 
        # item = [2,0] data point 
        # tree = dict 
        # checks if item is in tree and returns True or False 
  
        if "data" in tree: 
            # print("data", tree)

            #list_ = json.loads(tree["data"])
            
            #if np.array_equal(list_,list(item)): 
            #BEFORE
            if np.array_equal(list(tree["data"]),list(item)): 
                return True 

        else: 
            allNodes = tree['nodes']
            for node in allNodes: 

                if self.find_item_in_tree(item, node) == True: 

                    return True 

        return False 






        


 

        
        
    



            
             



        
 

 