

PIPELINE_CONFIG_PATH=/home/johnny/twitter_pride_vanity/algorithms/face_detection_config/faster_rcnn_resnet101_voc07.config

MODEL_DIR=/home/johnny/RCNN/temp_train
EXPORTED_MODEL_DIR=/home/johnny/RCNN/temp_final

SAMPLE_1_OF_N_EVAL_EXAMPLES=1

TRAIN_DATASET=/home/johnny/RCNN/dataset/WIDER_train.record
VAL_DATASET=/home/johnny/RCNN/dataset/WIDER_val.record

TRAIN_DATAROOT=/home/johnny/RCNN/dataset/WIDER_train/images
VAL_DATAROOT=/home/johnny/RCNN/dataset/WIDER_val/images

TRAIN_ANNOTATION=/home/johnny/RCNN/dataset/wider_face_split/wider_face_train_bbx_gt.txt
VAL_ANNOTATION=/home/johnny/RCNN/dataset/wider_face_split/wider_face_val_bbx_gt.txt

python3 /home/johnny/twitter_pride_vanity/algorithms/face_detection_dataset/WIDER_FACE_TFRecord.py\
    ${TRAIN_DATAROOT}\
    ${TRAIN_ANNOTATION}\
    ${TRAIN_DATASET}

python3 /home/johnny/twitter_pride_vanity/algorithms/face_detection_dataset/WIDER_FACE_TFRecord.py\
    ${VAL_DATAROOT}\
    ${VAL_ANNOTATION}\
    ${VAL_DATASET}


# Train
rm  -rf ${MODEL_DIR}/*


python3 /home/johnny/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr


#Export
rm  -rf ${EXPORTED_MODEL_DIR}/*
python3 /home/johnny/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${MODEL_DIR}/model.ckpt-4066 \
    --output_directory ${EXPORTED_MODEL_DIR}


