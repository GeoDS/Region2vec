import os
from analytics import run_aggclustering 
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n_clusters', type=int, default=14,
                    help='Number of clusters.')
parser.add_argument('--affinity', type=str, default='euclidean',
                    help='affinity metric')
parser.add_argument('--filename', type=str, default='Epoch_378_lr_0.001_dropout_0.1_hidden_16_output_14_patience_50_hos_5.0_losstype_div.csv',
                    help='file name')
                  
args = parser.parse_args()


if '../' in args.filename:
    args.filename = args.filename.split('/')[-1]

linkage = 'ward'
path = '../result/'

labels, total_ratio, median_ineq, median_cossim, median_dist, homo_score = run_aggclustering(path, args.filename, args.affinity, args.n_clusters, linkage)
csv_data = [args.filename, args.n_clusters, linkage, args.affinity, total_ratio, median_ineq, median_cossim, median_dist, homo_score]

result_csv = 'cluster_result.csv'

if not os.path.exists(os.path.join(path, result_csv)):
    with open(os.path.join(path, result_csv), 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['file_name', 'n_clusters', 'linkage', 'distance', 'total_ratio', 'median_ineq', 'median_cossim','median_dist', "homo_score"]
        csv_write.writerow(csv_head)
        f.close()

with open(os.path.join(path, result_csv), 'w') as f:
    csv_write = csv.writer(f)
    csv_write.writerow(csv_data)
    f.close()