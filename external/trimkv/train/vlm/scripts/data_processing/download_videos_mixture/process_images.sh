export $(cat .env | xargs)
echo "DATASET_DIR is set to '$DATASET_DIR'"
IMAGE_SRC_DIR="${DATASET_DIR}/M4-Instruct-Data"
IMAGE_DEST_DIR="${DATASET_DIR}/m4-instruct/images"
echo "IMAGE_SRC_DIR is set to '$IMAGE_SRC_DIR'"
echo "IMAGE_DEST_DIR is set to '$IMAGE_DEST_DIR'"

mkdir -p ${IMAGE_DEST_DIR}
cp ${IMAGE_SRC_DIR}/m4_instruct_annotations.json ${IMAGE_DEST_DIR}/m4_instruct_annotations.json
# following this issue: https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data#:~:text=For%20dreamsim_split.z01%20and%20dreamsim_split.zip%2C%20please%20run%20%22zip%20%2Ds%200%20dreamsim_split.zip%20%2D%2Dout%20dreamsim.zip%22 
if [[ ! -f "${IMAGE_SRC_DIR}/dreamsim.zip" ]]; then
    zip -s 0 "${IMAGE_SRC_DIR}/dreamsim_split.zip" --out "${IMAGE_SRC_DIR}/dreamsim.zip"
else
    echo "Skipping merge (dreamsim.zip already exists)"
fi

# Unzip all image datasets
for file in ${IMAGE_SRC_DIR}/*.zip; do
    if [[ "$file" == "$IMAGE_SRC_DIR/dreamsim_split.zip" || "$file" == "$IMAGE_SRC_DIR/dreamsim_split.z01" ]]; then
        echo "Skipping DreamSim split file: $file"
        continue
    fi
    name="${file%.zip}"
    outdir="$IMAGE_DEST_DIR/$name"

    if [[ -d "$outdir" ]]; then
        echo "Skipping $file (already extracted)"
        continue
    fi

    mkdir -p "$outdir"
    unzip -q "$file" -d "$IMAGE_DEST_DIR"
done

python3 scripts/data/download_videos_mixture/fix_annotations.py -j ${IMAGE_DEST_DIR}/m4_instruct_annotations.json

