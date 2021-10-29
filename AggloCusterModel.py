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
        
    def addNode(self, distance, index):
        pass
    
    def findClosestPair(self):
        pass
    
    def 