#!/bin/bash

TARGET_CLIP_DURATION=9
USER_DIRECTORY=/scratch/groups/rwr/joswong/CS230/VM-NET/data/user

source ../load.sh

# input file stored in <data/user>
input_file=$USER_DIRECTORY/${1}

# get duration of the video clip
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file")
echo "$input_file is $DURATION seconds long"
echo "${input_file/.mp4/_clipped.mp4}"
# clip video to TARGET_CLIP_DURATION
if (( $(echo "$DURATION >= $TARGET_CLIP_DURATION" | bc -l) )); then
    echo "Clipping the first $TARGET_CLIP_DURATION seconds of $input_file..."
    ffmpeg -y -ss 0 -t "$TARGET_CLIP_DURATION" -i "$input_file" -c copy -copyinkf "${input_file/.mp4/_clipped.mp4}"
fi

# extract video only (no audio)
ffmpeg -i ${input_file/.mp4/_clipped.mp4} -c copy -an "${input_file/.mp4/_clipped_video.mp4}"

# record in .csv file for Youtube8M feature extracion 
> $USER_DIRECTORY/record.csv
echo "${input_file/.mp4/_clipped_video.mp4},1" >> $USER_DIRECTORY/record.csv

# run youtube embedding
python3 ../youtube-8m/feature_extractor/extract_tfrecords_main.py --output_tfrecords_file $USER_DIRECTORY/tfrecords --input_videos_csv $USER_DIRECTORY/record.csv

ml reset
source ../load_net.sh

# get best clip
python test.py user ${input_file/.mp4/_clipped_video_feature.pkl}

# collate music and video
ffmpeg -i ${input_file/.mp4/_clipped_video.mp4} -i ${input_file/.mp4/_clipped_video.mp3} -shortest -c:v copy -c:a aac ${input_file/.mp4/_clipped_video_final.mp4}

# move intermediate files to repository
mkdir $USER_DIRECTORY/auxil_${1/.mp4//}
mv ${input_file/.mp4/_clipped.mp4} $USER_DIRECTORY/auxil_${1/.mp4//}
mv ${input_file/.mp4/_clipped_video.mp4} $USER_DIRECTORY/auxil_${1/.mp4//}
mv ${input_file/.mp4/_clipped_video_feature.pkl} $USER_DIRECTORY/auxil_${1/.mp4//}
mv ${input_file/.mp4/_clipped_video.mp3} $USER_DIRECTORY/auxil_${1/.mp4//}
