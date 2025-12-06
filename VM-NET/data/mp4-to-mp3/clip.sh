#! /bin/bash

video='/scratch/groups/rwr/joswong/CS230/VM-NET/data/raw'
video_clipped='/scratch/groups/rwr/joswong/CS230/VM-NET/data/video'

CLIP_DURATION=10
CLIP_DURATION_LONG=15

num=1
> video.csv

for f in $(find $video -type f -name "*.mp4");
   do  
      DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
      OUTPUT_FILE=${video_clipped}/$(printf "%03d" "$num").mp4
     
      # Print the total duration
      echo "$f is $DURATION seconds long"

      # Check if duration is greater than or equal to CLIP_DURATION
      if (( $(echo "$DURATION >= $CLIP_DURATION" | bc -l) )); then
          echo "Clipping the first $CLIP_DURATION seconds of $f..."
          ffmpeg -y -ss 0 -t "$CLIP_DURATION" -i "$f" -c copy -copyinkf "${OUTPUT_FILE/.mp4/_frnt.mp4}"
       
          # Record in .csv file
          echo "${OUTPUT_FILE/.mp4/_frnt.mp4},$(basename ${f%.*})" >> video.csv
      fi 

      # Check if duration is greater than CLIP_DURATION_LONG
      if (( $(echo "$DURATION >= $(($CLIP_DURATION_LONG))" | bc -l) )); then
          echo "Clipping the last $CLIP_DURATION seconds of $f..."
          ffmpeg -y -sseof -${CLIP_DURATION} -i "$f" -c copy -copyinkf "${OUTPUT_FILE/.mp4/_back.mp4}"

          # Record in .csv file
          echo "${OUTPUT_FILE/.mp4/_back.mp4},$(basename ${f%.*})" >> video.csv
      fi 

      num=$((num+1)) 
   done


