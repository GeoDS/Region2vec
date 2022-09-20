#!/bin/bash 

source D:/Applications/anaconda3/etc/profile.d/conda.sh
conda activate D:/Applications/envs/gcn

#ltype div/log #output 8 13
#for hops in 1 5
#do
python train.py --ltype div --hops 5 --output 14 --hidden 16 
    #python train.py --ltype div --hops $hops --output 13 --hidden 32
#done