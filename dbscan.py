import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from DBscanModel import DBScanModel
from kmeans import printClusters

calcSSE = lambda cluster: np.sum((np.array(cluster) - np.mean(cluster, axis=0)) ** 2)
#java dbscan <Filename> <epsilon> <NumPoints>
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


    args = vars(parser.parse_args())
    return args

def genClusterData(cluster):
    if(len(cluster) == 0):
        return ""
    centroid = np.mean(cluster, axis=0)
    distances = np.sqrt(np.sum((np.array(cluster) - np.array(centroid)) ** 2, axis=0))
    cluster = np.array(cluster).tolist()
    return (f'\tCenter: {", ".join([str(x) for x in centroid.tolist()])}\n' +
    f'\tMax Dist. to Center: {str(distances.max())}\n' +
    f'\tMin Dist. to Center: {str(distances.min())}\n' +
    f'\tAvg Dist. to Center: {str(distances.mean())}\n' + 
    f'\t{str(len(cluster))} Points:\n\t\t' +
    "\n\t\t".join([", ".join([str(x) for x in point]) for point in cluster]))

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
    model = DBScanModel(data, radius, minPoints)
    clusters = model.build()
    outliers = [i for i in range(len(data)) if model.type.get(i) is None]
    SSE = np.sum([calcSSE(cluster) for cluster in clusters])

    print("\n\n".join([f'Cluster {i}:\n {genClusterData(data[(model.clusters[i])])}' for i in range(len(model.clusters))]))
    print(f'Outliers: {data[(outliers)]}')
    print(f'Total SSE: {SSE}')
    
    if len(data[0]) == 2:
        if len(outliers) > 0:
            clusters.append(data[(outliers)])
        printClusters(clusters)
    

if __name__ == "__main__":
    main()
