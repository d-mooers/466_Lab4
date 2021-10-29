import pandas as pd
import numpy as np

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))
    
class DBScanModel:
    def __init__(self, data, radius, minPoints):
        self.data = data
        self.distances = [np.apply_along_axis(distanceFrom(point), 1, data) for point in data]
        self.radius = radius
        self.minPoints = minPoints
        self.visited = {}
        self.type = {}
        
    # Iterate through the distance matrix, a point is core iff it has X edges that are less than radius, s.t. X >= minPoints
    def findCorePoints(self):
        pass
    
    # Performs DFS on the distance matrix, where non-core points are treated as leaf nodes
    def densityConnectivity(self, currentIndex, clusterNumber):
        pass
    
    # Iterates through all points, and a point is boundary iff it is not a core point and it was visited
    def findBoundaryPoints(self):
        pass
    
    # Iterates through all points, and a point is outlier iff it is not a core point and it was NOT visited
    def findOutliers(self):
        pass
    
    # 1 - find the core points
    # 2 - densityConnectivity for all core points
    # 3 - find the boundary points
    # 4 - find the outliers
    def build(self):
        pass