#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 3
# Dylan Mooers - dmooers@calpoly.edu
# Justin Huynh - jhuynh42@calpoly.edu
# Purpose - Builds the underlying tree for C45 and outputs to JSON

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from KMeansModel import KMeans
from matplotlib import pyplot as plt
import json

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))

def printClusters(clusters):
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'teal', 'brown', 'black']
    for i in range(len(clusters)):
        cluster = clusters[i]
        for [x, y] in cluster:
            plt.plot(x, y, colors[i % len(colors)], marker='o')
    plt.show()
    
def calcInterCentroidDistance(centroids):
    return np.sum(np.apply_along_axis(distanceFromAll(centroids), 1, centroids))

def outputClusterData(clusters, centroids):
    SSE = np.sum([genClusterData(clusters[i], i, centroids[i]) for i in range(len(clusters))])
    interCentroidDistance = calcInterCentroidDistance(centroids) / (len(centroids) ** 2)
    print(f'SSE: {SSE}')
    print(f'Inter Centroid Distance: {interCentroidDistance}')
    print(f'SSE / Inter Centroid Distance: {SSE / interCentroidDistance}')

    
    
def genClusterData(cluster, clusterNumber, centroid):
    if centroid is None:
        centroid = centroid.mean(axis=0)
    distances = np.sqrt(np.sum((np.array(cluster) - np.array(centroid)) ** 2, axis=0))
    cluster = np.array(cluster).tolist()
    print(f'Cluster {clusterNumber}:')
    print(f'\tCenter: {", ".join([str(x) for x in centroid.tolist()])}\n' +
    f'\tMax Dist. to Center: {str(distances.max())}\n' +
    f'\tMin Dist. to Center: {str(distances.min())}\n' +
    f'\tAvg Dist. to Center: {str(distances.mean())}\n' + 
    f'\t{str(len(cluster))} Points:\n\t\t' +
    "\n\t\t".join([", ".join([str(x) for x in point]) for point in cluster]))
    return np.sum((np.array(cluster) - np.array(centroid)) ** 2) / len(cluster)

#java kmeans <Filename> <k>
def parse():
    parser = argparse.ArgumentParser(description="C45")
    parser.add_argument(
        "trainingSetFile", type=str, help="name of csv file containing dataset"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="?",
        default=3,
        help="integer value reresenting number of neighbors; default is 3",
    )
    parser.add_argument(
        "--t",
        type=float,
        nargs="?",
        default=0.1,
        help="float value reresenting threshold; default is 0.1",
    )
    parser.add_argument("--change", default=False, action="store_true", help="use centroid change as stoppage condition")


    args = vars(parser.parse_args())
    return args


def main():
    args = parse()
    training_fname = args["trainingSetFile"]
    threshold = args["t"]
    k = args['k']
    useSSE = not args['change']

    tmp = pd.read_csv(training_fname)

    header = list(tmp)
    skip = [i for i, j in zip(tmp.columns, header) if j == '0']
    tmp.drop(tmp.index[[0]], inplace=True)
    tmp.drop(columns=skip, inplace=True)
    skip2 = tmp[(tmp == '?').any(axis=1)]
    tmp.drop(skip2.index, axis=0,inplace=True)
    tmp.reset_index(drop=True, inplace=True)
    data = np.array(tmp)
    model = KMeans(data, k, threshold, useSSE=useSSE)
    model.run()
    outputClusterData(model.clusters, model.centroids)
    if len(data[0]) == 2:
        printClusters(model.clusters)
    

if __name__ == "__main__":
    main()
