#!/bin/bash
for i in ./Samples/*.*
do
   echo $i 
   python2.7 main_load.py $i
done