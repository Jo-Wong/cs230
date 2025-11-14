#!/bin/bash

> train.txt

for f in video_clipped/*; do
   
   if [[ "$f" == *".mp4"* ]]; then
      echo $(basename ${f%.*}) >> train.txt
   fi

done
