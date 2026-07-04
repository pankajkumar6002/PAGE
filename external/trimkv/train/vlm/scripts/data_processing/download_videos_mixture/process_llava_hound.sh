export $(cat .env | xargs)
echo "DATASET_DIR: ${DATASET_DIR}"
VIDEO_SRC_DIR="${DATASET_DIR}/hf_videos"
echo "VIDEO_SRC_DIR: ${VIDEO_SRC_DIR}"

cur_dir=$(pwd)
cd ${VIDEO_SRC_DIR}/llava_hound/train_300k

tar -xvf chunk_0.tar.gz
tar -xvf chunk_1.tar.gz
tar -xvf chunk_2.tar.gz
tar -xvf chunk_3.tar.gz
tar -xvf chunk_4.tar.gz
tar -xvf chunk_5.tar.gz
tar -xvf chunk_6.tar.gz
tar -xvf chunk_7.tar.gz
tar -xvf chunk_8.tar.gz
tar -xvf chunk_9.tar.gz
tar -xvf chunk_10.tar.gz
tar -xvf chunk_11.tar.gz
tar -xvf chunk_12.tar.gz
tar -xvf chunk_13.tar.gz
tar -xvf chunk_14.tar.gz
tar -xvf chunk_15.tar.gz

cd $cur_dir

FPS=8 bash llava_hound_frames2mp4.sh
python3 fix_annotations_llavahound.py -j ${DATASET_DIR}/hf_videos/llava_hound/sharegptvideo_qa_255k_processed.json
