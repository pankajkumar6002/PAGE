export $(cat .env | xargs)

mkdir -p ${DATASET_DIR}/debug_data
wget https://huggingface.co/datasets/JunHill/llava_data/resolve/main/debug_data.zip?download=true -O ${DATASET_DIR}/debug_data/debug_data.zip
unzip -o ${DATASET_DIR}/debug_data/debug_data.zip -d ${DATASET_DIR}/
