import pandas as pd
import numpy as np

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))
    
class KMeans:
    
    def __init__(self, data, k, threshold):
        self.data = data
        self.k = k
        self.threshold = threshold
        self.centroids = np.array()
        self.clusters = [[] for _ in range(k)]
        
    # returns the error of centroid selection
    def findNearestCentroid(self, row):
        distances = np.apply_along_axis(distanceFrom(row), 1, self.centroids)
        minDistanceIndex = np.argmin(distances)
        self.clusters[minDistanceIndex].append(row)
        return distances[minDistanceIndex]
    
    # Returns the SSE of this generation of clusters
    def generateClusters(self):
        return np.sum(np.apply_along_axis(self.findNearestCentroid, 1, self.data) ** 2)
    
    def canStop(self, SSE):
        return SSE <= self.threshold
    
    # returns the change in centroids
    def calculateNewCentroids(self):
        oldCentroids = self.centroids
        self.centroids = np.array(self.clusters.map(lambda cluster: np.mean(np.array(cluster), axis=1)))
        return np.sum(np.sqrt(np.sum((oldCentroids - self.centroids) ** 2, axis=1), axis= 1))
    
    def getFarthestPointFromCentroids(self):
        distances = np.apply_along_axis(distanceFromAll(np.array(self.centroids)), 1, self.data)
        return self.data[np.argmax(distances)]
        
    def kmeansPlus(self):
        self.centroids = [np.mean(self.data, axis=1)]
        for _ in range(self.k - 1):
            self.centroids.append(self.getFarthestPointFromCentroids())
    
    def run(self):
        self.kmeansPlus()
        SSE = self.threshold + 1
        oldClusters = None
        oldCentroids = None
        while (not self.canStop(SSE)):
            SSE = self.generateClusters()
        return oldClusters
    
        
    