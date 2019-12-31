#!/bin/sh

if [ "$#" -ne 4 ]; then
echo "Usage: $0 "
exit
fi

url=$1
chunk_size=$2
sample_rate=$3
dataset_path=$4

converted="joint2.wav"
ffmpeg -i $url -ac 1 -ab 16k -ar $sample_rate $converted

mkdir $dataset_path
length=$(ffprobe -i $converted -show_entries format=duration -v quiet -of csv="p=0")
end=$(echo "$length / $chunk_size - 1" | bc)
echo "splitting..."
for i in $(seq 0 $end); do
ffmpeg -hide_banner -loglevel error -ss $(($i * $chunk_size)) -t $chunk_size -i $converted "$dataset_path/$i.wav"
done
echo "done"
rm -f $converted
