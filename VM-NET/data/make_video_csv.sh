#!/bin/bash

num=0

for f in video_clipped/*; do
   
   num=$((num+1))

   if [[ "$f" == *".mp4"* ]]; then
      if [[ $num == 1 ]]; then
         echo "../../VM-NET/data/video_clipped/$(basename $f),${num}" > video_clipped.csv
      else
         echo "../../VM-NET/data/video_clipped/$(basename $f),${num}" >> video_clipped.csv
      fi
   fi

done
