CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}"
PASCAL_FOLDER="pascal_voc_seg"
EXP_FOLDER="exp/train_on_trainval_set"
PASCAL_DATASET="/home/ubuntu/Desktop/image_resized1/tfrecord"
TRAIN_LOGDIR="/home/ubuntu/Desktop/models-master/research/deeplab/colorectal/logs"
VIS_LOGDIR="${WORK_DIR}/visfull111"
python vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=400 \
  --vis_crop_size=400 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}"\
  --max_number_of_iterations=1
