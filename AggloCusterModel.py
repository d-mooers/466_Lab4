import pandas as pd
import numpy as np

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
        self.clusters = set() # set of tuples s.t. (0, 1, 2) -> data[0],data[1],data[2] are in the same cluster
        
    # Adds the given node to the tree
    def addNode(self, distance, index):
        pass
    
    # Finds the closest clusters, taking into account the link method and 
    # already existing clusters
    def findClosestPair(self):
        pass
    
    # Iteratively finds the closest clusters and then creates a new cluster
    # Stops when there is only one cluster
    def build(self):
        pass