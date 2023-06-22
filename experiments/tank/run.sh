#!/bin/sh

python ../../run.py $1 tank --num-parallel 100000 --horizon 20 --epochs 150 --mini-epochs 1 --exp-name vanilla_rl
