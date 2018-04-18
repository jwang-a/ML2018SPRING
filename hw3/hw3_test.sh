#!/bin/bash
wget -O model.plk https://www.dropbox.com/s/j4fm92lyp6z9qzj/model.plk?dl=1
python3 test.py $1 $2
