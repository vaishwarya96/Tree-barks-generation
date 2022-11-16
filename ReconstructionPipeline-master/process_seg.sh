# Beware this script can take days to run

videos=$1
output=$2

for i in $(ls mask_dataset/images);
do
    python3 seg_postprocessing.py --tree_id $i --images_path mask_dataset/images --mask_path NN_masks --output_path final_masks &
done

FAIL=0
for job in `jobs -p`
do
    echo $job
    wait $job || let "FAIL+=1"
done

