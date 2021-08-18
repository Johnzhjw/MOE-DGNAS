#!/bin/bash
module load anaconda/2020.11 
source activate graphlab
python citation.py --dataset Cora --gpu 0
