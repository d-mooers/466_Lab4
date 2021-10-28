import pandas as pd
import numpy as np

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton))
    
class KMeans:
    def __init__(self, data, k, threshold):
        self.data = data
        self.k = k
        self.threshold = threshold
        self.centroids = np.array()
        self.clusters = [[] for _ in range(k)]
        
    # returns the error of centroid selection
    def findNearestCentroid(self, row):
        distances = np.apply_along_axis(distanceFrom(row), axis=1)
        minDistanceIndex = np.argmin(distances)
        self.clusters[minDistanceIndex].append(row)
        return distances[minDistanceIndex]
    
    # Returns the SSE of this generation of clusters
    def generateClusters(self):
        return np.sum(np.apply_along_axis(self.findNearestCentroid, 1) ** 2)
    
    def canStop(self, SSE):
        return SSE <= self.threshold
    
    # returns the change in centroids
    def calculateNewCentroids(self):
        oldCentroids = self.centroids
        self.centroids = np.array(self.clusters.map(lambda cluster: np.mean(np.array(cluster), axis=1)))
        return np.sum(np.sqrt(np.sum((oldCentroids - self.centroids) ** 2, axis=1), axis=1))
    
    def kmeansPlus(self):
        
    
    def run(self):
        
    
        
    