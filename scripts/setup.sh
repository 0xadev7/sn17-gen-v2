# Base server
conda env create -f env/environment.yml
conda activate sn17-gen-v2

# Trellis
cd scripts/trellis
bash setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# Back to root
cd ../..
pip install -r requirements.txt

# Download SyncDreamer
python3 -m pip install --upgrade gdown
mkdir gen/lib/sync_dreamer/ckpt
gdown --fuzzy "https://drive.google.com/file/d/1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam/view?usp=sharing" \
-O gen/lib/sync_dreamer/ckpt/syncdreamer.ckpt
