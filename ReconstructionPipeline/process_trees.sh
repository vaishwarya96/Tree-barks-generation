# Beware this script can take days to run

videos=$1
output=$2
nn_path=$3

mkdir $output/mask_dataset
for video in $(ls $videos)
do
    name=$(echo $video | cut -d. -f1)
    echo $name
    python3 OpticalFlow.py --video_path $videos/$video --tree_id $name --output $output/mask_dataset &
done

FAIL=0
for job in `jobs -p`
do
   echo $job
   wait $job || let "FAIL+=1"
done

for i in $(ls mask_dataset/*_full.txt);
do 
    cat $i >> masking_description.txt
    rm -rf $i
done
for i in $(ls mask_dataset/*.txt);
do 
    cat $i >> dataset_description.txt
    rm -rf $i
done
