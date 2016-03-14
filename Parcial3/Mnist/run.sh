#!/bin/bash
for i in {0..9}
do
   echo "Image $i :"
   python2.7 main_load.py $i
done