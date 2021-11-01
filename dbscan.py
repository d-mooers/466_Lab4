import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from DBscanModel import DBScanModel
from kmeans import printClusters

#java dbscan <Filename> <epsilon> <NumPoints>
def parse():
    parser = argparse.ArgumentParser(description="DB Scan")
    parser.add_argument(
        "trainingSetFile", type=str, help="name of csv file containing dataset"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="?",
        default=0.1,
        help="integer value reresenting number representing epsilon; default is __",
    )

    parser.add_argument(
        "--NumPoints",
        type=int,
        nargs="?",
        default=5,
        help="integer value reresenting number representing number of points; default is __"
    )


    args = vars(parser.parse_args())
    return args


def main():
    args = parse()
    training_fname = args["trainingSetFile"]
    radius = args["epsilon"]
    minPoints = args["NumPoints"]


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
    print(model.type)
    print(clusters)
    # printClusters(clusters)
    

if __name__ == "__main__":
    main()
