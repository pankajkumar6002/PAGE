mkdir -p data/llava_data/data/textvqa/
mkdir -p data/llava_data/data/coco/
mkdir -p data/llava_data/data/gqa/
mkdir -p data/llava_data/data/vg/VG_100K/
mkdir -p data/llava_data/data/vg/VG_100K_2/

wget http://images.cocodataset.org/zips/train2017.zip -P data/llava_data/ 
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -P data/llava_data/
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P data/llava_data/
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P data/llava_data/
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P data/llava_data/

unzip data/llava_data/train2017.zip -d data/llava_data/data/coco/
unzip data/llava_data/images.zip -d data/llava_data/data/gqa/
unzip data/llava_data/train_val_images.zip -d data/llava_data/data/textvqa/
unzip data/llava_data/images.zip -d data/llava_data/data/vg/VG_100K/
unzip data/llava_data/images2.zip -d data/llava_data/data/vg/VG_100K_2/

mkdir -p data/llava_data/data/ocr_vqa
hf download JunHill/ocr-vqa --repo-type dataset --local-dir data/llava_data/data/ocr_vqa
cd data/llava_data/data/ocr_vqa 
python3 loadDataset.py
cd ../../../..