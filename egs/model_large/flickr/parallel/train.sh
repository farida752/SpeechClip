echo "[Train] SpeechCLIP Parallel Large on Flickr8k"
EXP_ROOT="exp_test"
CFG="config/speechCLIP/model_large/flickr/spchclp_p.yaml"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --config $CFG \
    --gpus 1 \
    --njobs 4 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT


