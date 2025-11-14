#! /bin/bash

music='/scratch/groups/rwr/joswong/VM-NET/data/audio/'
video='/scratch/groups/rwr/joswong/VM-NET/data/video/'
video_clipped='/scratch/groups/rwr/joswong/VM-NET/data/video_clipped/'

CLIP_DURATION=10

for f in $(find $video -type f -name "*.mp4");
   do  
      DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
      OUTPUT_FILE=$video_clipped/$(basename $f)

      # Check if duration is greater than or equal to CLIP_DURATION
      if (( $(echo "$DURATION >= $CLIP_DURATION" | bc -l) )); then
          echo "$f is $DURATION seconds long"
          echo "Clipping the first $CLIP_DURATION seconds of $f..."
          ffmpeg -y -ss 0 -t "$CLIP_DURATION" -i "$f" -c copy -copyinkf "$OUTPUT_FILE"
          echo "Clipped video saved to $OUTPUT_FILE"
      else
          echo "$f is shorter than $CLIP_DURATION seconds. No clipping performed."
      fi 
   done

