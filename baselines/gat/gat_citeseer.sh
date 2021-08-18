#!/bin/bash
module load anaconda/2020.11 
source activate graphlab
python train.py --dataset citeseer --gpu 0
