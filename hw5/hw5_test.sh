#!/bin/bash
wget -O WORD2VECDICT https://www.dropbox.com/s/8mpyv18m6w3nga1/WORD2VECDICT?dl=1
wget -O WORD2VECDICT.syn1neg.npy https://www.dropbox.com/s/xn533afy1il77y9/WORD2VECDICT.syn1neg.npy?dl=1
wget -O WORD2VECDICT.wv.syn0.npy https://www.dropbox.com/s/adcyw9r2pkrk6gp/WORD2VECDICT.wv.syn0.npy?dl=1
python3 train.py $1 $2
