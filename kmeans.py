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

def printClusters(clusters):
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'teal', 'brown', 'black']
    for i in range(len(clusters)):
        cluster = clusters[i]
        for [x, y] in cluster:
            plt.plot(x, y, colors[i], marker='o')
    plt.show()


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


    args = vars(parser.parse_args())
    return args


def main():
    args = parse()
    training_fname = args["trainingSetFile"]
    threshold = args["t"]
    k = args['k']

    tmp = pd.read_csv(training_fname)

    header = list(tmp)
    skip = [i for i, j in zip(tmp.columns, header) if j == '0']
    tmp.drop(tmp.index[[0]], inplace=True)
    tmp.drop(columns=skip, inplace=True)
    skip2 = tmp[(tmp == '?').any(axis=1)]
    tmp.drop(skip2.index, axis=0,inplace=True)
    tmp.reset_index(drop=True, inplace=True)
    data = np.array(tmp)
    model = KMeans(data, k, threshold)
    model.run()
    print(model.clusters)
    printClusters(model.clusters)
    

if __name__ == "__main__":
    main()
