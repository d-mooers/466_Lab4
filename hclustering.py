import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from AggloCusterModel import AgloClusterModel
from csv import reader
import json 
from kmeans import outputClusterData, pairWithData

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

def main():
    args = parse()
    training_fname = args["trainingSetFile"]
    threshold = args["t"]

    #training_fname = "medium_test.csv" #args["trainingSetFile"]
    #threshold = 1 #args["t"]

    

    tmp = pd.read_csv(training_fname)

    header = list(tmp)
    skip = [i for i, j in zip(tmp.columns, header) if j == '0']
    include = [i for i, j in zip(tmp.columns, header) if j != '0']

    # tmp.drop(tmp.index[[0]], inplace=True)
    full = tmp.copy()
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

    
    A = AgloClusterModel(data,str_data, threshold, full, include) 
    A.build()
    # print(json.dumps(A.tree, indent=2))
    clusters = A.measuring(A.threshold, A.tree)
    outputClusterData(clusters)
    # accuracy = np.sum([pairWithData(cluster, full, include) for cluster in clusters])
    # print(f'Accuracy: {accuracy}')

    # print(clusters, len(clusters))
    # A.testAllThresholds()
    # A.visualize()

    



main()