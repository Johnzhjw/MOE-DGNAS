#!/bin/bash
module load anaconda/2020.11 
source activate graphlab
python train.py --dataset pubmed --gpu 0
