#!/bin/bash
module load anaconda/2020.11 
source activate graphlab
python train_full.py --dataset citeseer --gpu 0 --aggregator-type mean
