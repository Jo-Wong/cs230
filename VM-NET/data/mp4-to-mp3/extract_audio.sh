#! /bin/bash

music='/scratch/groups/rwr/joswong/VM-NET/data/audio/'
video='/scratch/groups/rwr/joswong/VM-NET/data/video_clipped/'

for f in $(find $video -type f -name "*.mp4");
   do  
      ffmpeg -y -i $f -f wav -ab 160k -ar 44100 -ac 2 - | lame - "$music/$(basename ${f/mp4/mp3})"
   done

