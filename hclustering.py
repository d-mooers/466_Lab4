import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from AggloCusterModel import AgloClusterModel
from csv import reader
import json 
from kmeans import outputClusterData
from matplotlib import pyplot as plt

#olors = ["red", "yellow", "purple", "blue", "green", "pink", "brown", "black", "orange", "lightblue", "teal", "lightpurple", "tan", "lightgrey"]
colors = ["red", "yellow", "purple", "blue", "green", "pink", "brown", "black", "orange", "salmon", "teal", "violet", "lawngreen", "indigo"]
def parse():
    parser = argparse.ArgumentParser(description="HB Clustering")
    parser.add_argument(
        "trainingSetFile", type=str, help="name of csv file containing dataset"
    )
    parser.add_argument(
        "--t",
        type=float,
        nargs="?",
        default=3,
        help="float value reresenting threshold for stoppage condition; default is 3",
    )
    parser.add_argument("--normalize", default=False, action="store_true", help="normalize the data")
    parser.add_argument(
    "--distance",
    type=str,
    nargs="?",
    default="single",
    help="Distance between clusters, 'single', 'double' or 'average' "
    )

    args = vars(parser.parse_args())
    return args

def create_data_string(fname): 
    str_data = []
    with open(fname, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            json_string = json.dumps(row)
            str_data.append(json_string)
    return str_data


def printClusters(clusters, outliers , fname, t, dist_metric):
    for i in range(len(clusters)):
        cluster = clusters[i]
        for [x, y] in cluster:
            plt.title("Agglo: {} t = {} dist. metric = {}".format(fname, t, dist_metric))
            plt.plot(x, y, colors[i % len(colors)], marker='o')
    for [x, y] in outliers:
        plt.plot(x, y, "black", marker="x")
    plt.show()
    
def printClusters3d(clusters, outliers, fname, t, dist_metric):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(clusters)):
        cluster = clusters[i]
        # x = [c[0] for c in cluster]
        # y = [c[1] for c in cluster]
        # z = [c[2] for c in cluster]
        # print(list(zip(x, y, z)))
        for [x, y, z] in cluster:
            plt.title("Agglo: {} t = {} dist. metric = {}".format(fname, t, dist_metric))
            plt.plot(x, y,z, colors[i % len(colors)], marker='o')
    for [x, y, z] in outliers:
        plt.plot(x, y, z, "black", marker='x')
    # plt.xlim([-0.01, 1.1])
    # plt.ylim([-0.01, 1.1])

    plt.show()

def main():
    args = parse()
    training_fname = args["trainingSetFile"]
    threshold = args["t"]
    distance_metric = args["distance"]

    #training_fname = "medium_test.csv" #args["trainingSetFile"]
    #threshold = 1 #args["t"]

    

    tmp = pd.read_csv(training_fname)

    header = list(tmp)
    skip = [i for i, j in zip(tmp.columns, header) if j == '0']
    # tmp.drop(tmp.index[[0]], inplace=True)
    tmp.drop(columns=skip, inplace=True)
    skip2 = tmp[(tmp == '?').any(axis=1)]
    tmp.drop(skip2.index, axis=0,inplace=True)
    tmp.reset_index(drop=True, inplace=True)
    data = np.array(tmp)
    if args['normalize']:
        _min = data.min(axis=0)
        _max = data.max(axis=0)
        print(_min, _max)
        data = (data - _min) / (_max - _min)

    #print("data", data)
    str_data = create_data_string(training_fname)
    str_data = str_data[1:]
    #print('STR DATA', str_data)

    
    A = AgloClusterModel(data,str_data, threshold) 
    A.build(distance_metric)
    clusters = A.measuring(A.threshold, A.tree)
    outputClusterData(clusters)

    if len(data[0]) == 2:

        printClusters(clusters, [], training_fname, threshold, distance_metric)
    elif len(data[0]) == 3:
        printClusters3d(clusters, [], training_fname,  threshold, distance_metric)

    exit()
    
    A.testAllThresholds(training_fname, distance_metric)
    print(clusters, len(clusters))
    print("final clusters", A.final_clusters)
    A.visualize()

    



main()