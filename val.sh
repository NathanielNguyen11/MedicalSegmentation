CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
PASCAL_FOLDER="pascal_voc_seg"

#PASCAL_DATASET="/media/ubuntu/Quang/datset1/tfrecord"
PASCAL_DATASET="/home/ubuntu/Desktop/test_im/tfrecord"
#TRAIN_LOGDIR='/home/ubuntu/Desktop/models-master/research/deeplab/colorectal/logs'
TRAIN_LOGDIR="/home/ubuntu/Desktop/models-master/research/deeplab/logs1"
VIS_LOGDIR="${WORK_DIR}/evalfull1111"
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=2 \
  --eval_crop_size=400 \
  --eval_crop_size=400 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_evaluations=1

