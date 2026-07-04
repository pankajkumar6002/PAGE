export $(cat .env | xargs)
VIDEO_SRC_DIR="${DATASET_DIR}/hf_videos"

cur_dir=$(pwd)
cd ${VIDEO_SRC_DIR}/academic_v0_1/0_30_s_academic_v0_1

tar -xvf 0_30_s_academic_v0_1_videos_1.tar.gz
tar -xvf 0_30_s_academic_v0_1_videos_2.tar.gz
tar -xvf 0_30_s_academic_v0_1_videos_3.tar.gz
tar -xvf 0_30_s_academic_v0_1_videos_4.tar.gz
tar -xvf 0_30_s_academic_v0_1_videos_5.tar.gz
tar -xvf 0_30_s_academic_v0_1_videos_6.tar.gz
tar -xvf 0_30_s_academic_v0_1_videos_7.tar.gz
tar -xvf 0_30_s_academic_v0_1_videos_8.tar.gz

cd $cur_dir

python3 fix_annotations.py -j ${DATASET_DIR}/hf_videos/academic_v0_1/0_30_s_academic_v0_1/0_30_s_academic_oe_v0_1_qa_processed.json
python3 fix_annotations.py -j ${DATASET_DIR}/hf_videos/academic_v0_1/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed.json
