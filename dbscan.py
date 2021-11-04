import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from DBscanModel import DBScanModel
from matplotlib import pyplot as plt
from kmeans import outputClusterData

calcSSE = lambda cluster: np.sum((np.array(cluster) - np.mean(cluster, axis=0)) ** 2)

colors = ['red', 'blue', 'green', 'yellow', 'orange', 'teal', 'brown']

def printClusters(clusters, outliers):
    for i in range(len(clusters)):
        cluster = clusters[i]
        for [x, y] in cluster:
            plt.plot(x, y, colors[i % len(colors)], marker='o')
    for [x, y] in outliers:
        plt.plot(x, y, "black", marker="x")
    plt.show()
    
def printClusters3d(clusters, outliers):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(clusters)):
        cluster = clusters[i]
        # x = [c[0] for c in cluster]
        # y = [c[1] for c in cluster]
        # z = [c[2] for c in cluster]
        # print(list(zip(x, y, z)))
        for [x, y, z] in cluster:
            plt.plot(x, y,z, colors[i % len(colors)], marker='o')
    for [x, y, z] in outliers:
        plt.plot(x, y, z, "black", marker='x')
    # plt.xlim([-0.01, 1.1])
    # plt.ylim([-0.01, 1.1])

    plt.show()


def parse():
    parser = argparse.ArgumentParser(description="DB Scan")
    parser.add_argument(
        "trainingSetFile", type=str, help="name of csv file containing dataset"
    )

    parser.add_argument(
        "--e",
        type=float,
        nargs="?",
        default=0.1,
        help="integer value reresenting number representing epsilon; default is __",
    )

    parser.add_argument(
        "--n",
        type=int,
        nargs="?",
        default=5,
        help="integer value reresenting number representing number of points; default is __"
    )
    parser.add_argument("--normalize", default=False, action="store_true", help="normalize the data")

    args = vars(parser.parse_args())
    return args

# def genClusterData(cluster):
#     if(len(cluster) == 0):
#         return ""
#     centroid = np.mean(cluster, axis=0)
#     distances = np.sqrt(np.sum((np.array(cluster) - np.array(centroid)) ** 2, axis=0))
#     cluster = np.array(cluster).tolist()
#     return (f'\tCenter: {", ".join([str(x) for x in centroid.tolist()])}\n' +
#     f'\tMax Dist. to Center: {str(distances.max())}\n' +
#     f'\tMin Dist. to Center: {str(distances.min())}\n' +
#     f'\tAvg Dist. to Center: {str(distances.mean())}\n' + 
#     f'\t{str(len(cluster))} Points:\n\t\t' +
#     "\n\t\t".join([", ".join([str(x) for x in point]) for point in cluster]))

def main():
    args = parse()
    training_fname = args["trainingSetFile"]
    radius = args["e"]
    minPoints = args["n"]


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

    model = DBScanModel(data, radius, minPoints)
    clusters = model.build()
    outliers = [i for i in range(len(data)) if model.type.get(i) is None]
    # SSE = np.sum([calcSSE(cluster) for cluster in clusters])

    # print("\n\n".join([f'Cluster {i}:\n {genClusterData(data[(model.clusters[i])])}' for i in range(len(model.clusters))]))
    outputClusterData(clusters)
    print(f'Outliers: {data[(outliers)]}')
    # print(f'Total SSE: {SSE}')
    
    if len(data[0]) == 2:
        printClusters(clusters, data[(outliers)])
    elif len(data[0]) == 3:
        printClusters3d(clusters, data[(outliers)])

    

if __name__ == "__main__":
    main()
