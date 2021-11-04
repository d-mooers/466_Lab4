#!/usr/bin/env python3
# CSC 466
# Fall 2021
# Lab 3
# Dylan Mooers - dmooers@calpoly.edu
# Justin Huynh - jhuynh42@calpoly.edu
# Purpose - Builds the underlying tree for C45 and outputs to JSON

import argparse
from matplotlib.colors import cnames
import pandas as pd
import numpy as np
from pathlib import Path
from KMeansModel import KMeans
from matplotlib import pyplot as plt
import json

distance = lambda x,y: np.sqrt(np.sum((x - y) ** 2))
distanceFrom = lambda origin: lambda destinaton: distance(origin, destinaton)
distanceFromAll = lambda originPoints: lambda destination:  np.sum(np.apply_along_axis(distanceFrom(destination), 1, originPoints))

colors = ['red', 'blue', 'green', 'yellow', 'orange', 'teal', 'brown', 'black']

def printClusters(clusters):
    for i in range(len(clusters)):
        cluster = clusters[i]
        for [x, y] in cluster:
            plt.plot(x, y, colors[i % len(colors)], marker='o')
    plt.show()
    
def printClusters3d(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(clusters)):
        cluster = clusters[i]
        for [x, y, z] in cluster:
            plt.plot(x, y,z, colors[i % len(colors)], marker='o')
    # plt.xlim([-0.01, 1.1])
    # plt.ylim([-0.01, 1.1])

    plt.show()

def calcInterCentroidDistance(centroids):
    dist = 0
    iter = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist += distance(centroids[i], centroids[j])
            iter += 1
    # print(f'Dist: {dist}, Iter: {iter}')
    return dist / iter

def outputClusterData(clusters, centroids=None):
    if centroids is None:
        centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        
    total = 0
    for cluster in clusters: total += len(cluster)
    
    avgRadius = np.sum([genClusterData(clusters[i], i, centroids[i]) for i in range(len(clusters))])
    interCentroidDistance = calcInterCentroidDistance(centroids)
    print(f'Average Cluster Radius: {avgRadius}')
    print(f'Inter Centroid Distance: {interCentroidDistance}')
    print(f'Average Radius / Inter Centroid Distance: {avgRadius / interCentroidDistance}')
    return avgRadius / interCentroidDistance

    
    
def genClusterData(cluster, clusterNumber, centroid):
    if centroid is None:
        centroid = cluster.mean(axis=0)
    distances = np.sqrt(np.sum((np.array(cluster) - np.array(centroid)) ** 2, axis=1))
    cluster = np.array(cluster).tolist()
    print(f'Cluster {clusterNumber}:')
    print(f'\tCenter: {", ".join([str(x) for x in centroid.tolist()])}\n' +
    f'\tMax Dist. to Center: {str(distances.max())}\n' +
    f'\tMin Dist. to Center: {str(distances.min())}\n' +
    f'\tAvg Dist. to Center: {str(distances.mean())}\n' + 
    f'\t{str(len(cluster))} Points:\n\t\t' +
    "\n\t\t".join([", ".join([str(x) for x in point]) for point in cluster]))
    return distances.mean()

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
    parser.add_argument("--normalize", default=False, action="store_true", help="normalize the data")

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
    _min = 0
    _max = 0
    if args['normalize']:
        _min = data.min(axis=0)
        _max = data.max(axis=0)
        print(_min, _max)
        data = (data - _min) / (_max - _min)
        
    print("len data", len(data))
    ks = [k for k in range(2,len(data))]
    ratios = []
    
    for k in ks:
        print("k", k)
        model = KMeans(data, k, threshold, useSSE=useSSE)
        model.run()

        # if args['normalize']:
        #     minMax = _max - _min
        #     model.clusters = [minMax * cluster + _min for cluster in model.clusters]
        #     model.centroids = minMax * model.centroids + _min

        ratios.append(outputClusterData(model.clusters, model.centroids))
        # print("ratios", ratios)
        # if len(data[0]) == 2:
        #     printClusters(model.clusters)
        # elif len(data[0]) == 3:
        #     printClusters3d(model.clusters)

    print("ks", ks)
    print("ratios", ratios)
    xpoints = np.array(ks)
    ypoints = np.array(ratios)
    plt.title("SSE for K Means: {}".format(training_fname))
    plt.xlabel('K number of clusters')
    plt.ylabel('Ratio of: Avg. Radius / Inner Centroid Distance')

    plt.plot(xpoints, ypoints)
    plt.show()
    
    

if __name__ == "__main__":
    main()
