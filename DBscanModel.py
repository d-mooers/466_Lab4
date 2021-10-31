import pandas as pd
import numpy as np

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))
    
CORE = 0
BOUNDARY = 1
OUTLIER = 2
class DBScanModel:
    def __init__(self, data, radius, minPoints):
        self.data = data
        self.distances = np.array([np.apply_along_axis(distanceFrom(point), 1, data) for point in data])
        self.radius = radius
        self.minPoints = minPoints
        self.visited = {}
        self.type = {}
        self.clusters = {}
        
    # Iterate through the distance matrix, a point is core iff it has X edges that are less than radius, s.t. X >= minPoints
    def findCorePoints(self):
        neighbors = np.sum(self.distances <= self.radius, axis=0)
        areCorePoints = neighbors >= (self.minPoints + 1)
        corePoints = areCorePoints.nonZero()
        for id in corePoints:
            self.type[id] = CORE
        return corePoints
        
    
    # Performs DFS on the distance matrix, where non-core points are treated as leaf nodes
    def densityConnectivity(self, currentIndex, clusterNumber):
        if self.visited.get(currentIndex):
            return
        self.visited[currentIndex] = True
        self.clusters[clusterNumber].append(currentIndex)
        if self.type.get(currentIndex) is None:
            self.type[currentIndex] = BOUNDARY
        else:
            possibleNeighbors = self.distances[currentIndex]
            neighbors = possibleNeighbors <= self.radius
            for id in neighbors.nonZero():
                return self.densityConnectivity(id, clusterNumber)
    
    # Iterates through all points, and a point is outlier iff it is not a core point and it was NOT visited
    def findOutliers(self):
        outliers = np.array([i for i in range(len(self.distances)) if self.type.get(i) is None])
        alg = lambda i: self.type[i] = OUTLIER
        np.vectorize(alg)(outliers)
        
    
    # 1 - find the core points
    # 2 - densityConnectivity for all core points
    # 3 - find the boundary points
    # 4 - find the outliers
    def build(self):
        corePoints = self.findCorePoints()
        clusterNumber = 0
        clusters = []
        for point in corePoints:
            self.clusters[clusterNumber] = []
            self.densityConnectivity(point, clusterNumber)
            if len(self.clusters[clusterNumber]) > 0:
                clusterNumber += 1
                clusters.append(self.data[(self.clusters[clusterNumber])])
        return clusters
                
        