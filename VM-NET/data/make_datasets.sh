#!/bin/bash

SRC_DIR="audio_feature"
DST_DIR="/path/to/destination"
num_train=400
num_test=37

mkdir -p audio_feature/train
mkdir -p audio_feature/verification
mkdir -p audio_feature/test
mkdir -p video_feature/train
mkdir -p video_feature/verification
mkdir -p video_feature/test

output=$(find "$SRC_DIR" -maxdepth 1 -type f -print0 | shuf -z -n "$num_train")
output=(${output//.pkl/ })

for i in "${output[@]}"; do
   mv audio_feature/"$(basename $i).pkl" audio_feature/train/"$(basename $i).pkl"
   mv video_feature/"$(basename $i).pkl" video_feature/train/"$(basename $i).pkl" 
done

output=$(find "$SRC_DIR" -maxdepth 1 -type f -print0 | shuf -z -n "$num_test")
output=(${output//.pkl/ })

for i in "${output[@]}"; do
   mv audio_feature/"$(basename $i).pkl" audio_feature/test/"$(basename $i).pkl"
   mv video_feature/"$(basename $i).pkl" video_feature/test/"$(basename $i).pkl" 
done

for dtype in train test; do
   python3 make_datasets.py $dtype
done
