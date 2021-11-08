import numpy as np
import pandas as pd

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))
    

values = np.array([
    [8,5,3,4,5,6],
    [3,6,6,7,0,6],
    [3,8,5,1,2,9],
    [6,4,2,7,8,3]])

# values.min()          # = 1
# np.min(values)        # = 1
# np.amin(values)       # = 1
# print(min(values.flatten())) # = 1

# print("min= ", np.where(values == np.min(values)))
# print("min= ", type(np.where(values == np.min(values))))
# min_ = np.where(values == np.min(values))
# print("min", min_[0], min_[1])

clusters = np.array([[1], [2,3,4], [5]])

clusters = [np.apply_along_axis(distanceFrom(point), 1, values) for point in values]

from itertools import combinations
L = [[1], [2,3,4], [5]]

print([comb for comb in combinations(L, 2)])


# def go_through_trees(self, cluster): 
#     for tree in self.tree: 
#         return self.find_item_in_tree(tree)

    # searches list of trees and all trees for cluster 
        # if found 
            # removes the indv leaf or node from the tree list 
            # returns the node or leaf 
            
        # else 
            # returns a new leaf 
            # i.e { type:"leaf", "height":0, "data": "{}".format(self.data[c1])}

c1 = clusters[0] #[1]


data = [[1,0], [3,0], [6,0]]
tree = {"type": "root",
    "height": 5.0,
    "nodes": [{"type": "node",
                    "height": 2.0,
                    "nodes": [{ "type":"leaf", 
                                "height":0,
                                "data": [1,0]},
                                { "type": "leaf",
                                "height": 0,
                                "data" :[3,0]
                                }]},
                {"type": "leaf",
                "height": 0,
                "data": [6,0]}]}


# searches list of trees and all trees for cluster 
    # if found 
        # removes the indv leaf or node from the tree list 
        # returns the node or leaf 
        
    # else 
        # returns a new leaf 
        # i.e { type:"leaf", "height":0, "data": "{}".format(self.data[c1])}


# def find_item_in_tree(item, tree): 
    
#     if "data" in tree: 
#         # print("data", tree)
#         # print("tree[\"data\"]", tree["data"], item)
#         if tree["data"] == item: 
#             print("True")
#             return True 

#     else: 
#         allNodes = tree['nodes']
#         # print("all nodes", allNodes)
#         for node in allNodes: 
#             if find_item_in_tree(item, node) == True: 
#                 return True 

#     return False 
            


# print(find_item_in_tree([6,0], tree))