#!/bin/bash
module load anaconda/2020.11 
source activate graphlab
python train.py --dataset cora --gpu 0
