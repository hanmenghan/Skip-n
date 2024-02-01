replacements=("MiHO" "MiHIO" "MiHI" "None")

CUDA_VISIBLE_DEVICES=0 python eval_MiHIO.py --experiments greedy_fuyu --model fuyu-8b --beam 1 --MiHIO 

# Loop over each replacement
for replacement in "${replacements[@]}"; do
    # Execute the commands with the replacement
    CUDA_VISIBLE_DEVICES=0 python eval_MiHIO.py --experiments greedy_fuyu --model fuyu-8b --beam 1 --MiHIO ${replacement}
done


DIRECTORY='./log_beam/'
for SUBDIRECTORY in $DIRECTORY*/
do
  FOLDER_NAME=$(basename "$SUBDIRECTORY")
  for FILE in "$SUBDIRECTORY"*
  do
    echo "Processing file $FILE in folder: $FOLDER_NAME"
    python chair_eval.py --cap_file "$FILE"
  done
done
