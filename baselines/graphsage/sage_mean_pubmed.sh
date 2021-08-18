#!/bin/bash
module load anaconda/2020.11 
source activate graphlab
python train_full.py --dataset pubmed --gpu 0 --aggregator-type mean
