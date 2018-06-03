#!/bin/bash
wget -O WORD2VECDICT https://www.dropbox.com/s/8mpyv18m6w3nga1/WORD2VECDICT?dl=1
python3 test.py $1 $2
