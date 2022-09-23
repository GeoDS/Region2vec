# Region2vec

Region2vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions


## Paper

If you find our code useful for your research, please cite our paper:

*Liang, Y., Zhu, J., Ye, W., and Gao, S.\. (2022). [Region2vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions] ...* 

## Requirements

Region2 uses the following packages with Python 3.7

numpy==1.19.5

pandas==0.24.1

scikit_learn==1.1.2

scipy==1.3.1

torch==1.4.0



## Usage

1. run train.py to generate the embeddings.
```
python train.py
```
2. run clustering.py to generate the clustering result. 

```
python clustering.py --filename your_filename
```
Here the 'your_filename' should be replaced with the generated file from step 1.

3. Alternatively, to generate the clustering for all the files, please use bash, and run bash run_clustering.py.

```
bash run_clustering.sh 
```
Notes: the final results may vary depends on different platform and package versions.
The current result is obtained using Ubuntu with all pacakge versions in requirements.txt. 
