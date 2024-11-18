# Robust Fine-tuning of Zero-shot Models via Variance Reduction

This repository contains code for the Variance Reduction Fine-tuning (VRF).

**TLDR**: We propose a sample-wise ensembling technique for fine-tuning zero-shot models that can simultaneously attain the best ID and OOD accuracy.

## Install dependencies

```bash
conda env create --file environment.yml
conda activate vrf
```

## Add directory to PYTHONPATH:

```bash
cd VRF.public
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Prepare Datasets

### Prepare ImageNet

- (Skip if already downloaded) Download the dataset from the [official website](https://image-net.org/index.php) and extract to `${ANY_PATH_YOU_PREFER}/imagenet/images`.

- Create data directory and create symbolic links.
```bash
export DATA_LOCATION=~/data
cd $DATA_LOCATION
mkdir imagenet
ln -s ${ANY_PATH_YOU_PREFER}/imagenet/images imagenet
```

### Prepare ImageNet-V2

- (Skip if already downloaded) Download the dataset:
   ```bash
   cd ${ANY_PATH_YOU_PREFER}
   wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
   tar -xvf imagenetv2-matched-frequency.tar.gz
   rm imagenetv2-matched-frequency.tar.gz
   ```

- Create data directory and create symbolic links.
```bash
export DATA_LOCATION=~/data
cd $DATA_LOCATION
mkdir ImageNetV2-matched-frequency/
ln -s ${ANY_PATH_YOU_PREFER}/ImageNetV2-matched-frequency/ ImageNetV2-matched-frequency/
```

### Prepare ImageNet Sketch

- (Skip if already downloaded) Download the dataset from [Google Drive](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA)

- Create data directory and create symbolict links.
```bash
export DATA_LOCATION=~/data
cd $DATA_LOCATION
mkdir sketch/
ln -s ${ANY_PATH_YOU_PREFER}/imagenet-sketch/images/ sketch/
```

### Prepare ImageNet A

- (Skip if already downloaded):
```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xvzf imagenet-a.tar
rm imagenet-a.tar
```
- Create data directory and create symbolict links.
```bash
export DATA_LOCATION=~/data
cd $DATA_LOCATION
mkdir imagenet-a/
ln -s ${ANY_PATH_YOU_PREFER}/imagenet-adversarial/imagenet-a imagenet-a/
```

### Prepare ImageNet R

- (Skip if already downloaded):
```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvzf imagenet-r.tar
rm imagenet-r.tar
```

- Create data directory and create symbolict links.
```bash
export DATA_LOCATION=~/data
cd $DATA_LOCATION
mkdir imagenet-r 
ln -s ${ANY_PATH_YOU_PREFER}/imagenet-rendition/imagenet-r/ imagenet-r/
```

### Prepare ObjectNet
- (Skip if already downloaded):
```bash
wget https://objectnet.dev/downloads/objectnet-1.0.zip
unzip objectnet-1.0.zip
rm objectnet-1.0.zip
```

- Create data directory and create symbolict links.
```bash
export DATA_LOCATION=~/data
cd $DATA_LOCATION
mkdir objectnet-1.0
ln -s ${ANY_PATH_YOU_PREFER}/data3/data2/objectnet-1.0/ objectnet-1.0
```

## Download pre-trained models

- Download the available zero-shot and fine-tuned models (CLIP ViT-B/16) from [WiSE-FT](https://drive.google.com/drive/folders/1f56kjpRKPiNSaUxNDtETEDRkbDkZnpCQ?usp=sharing) to the directory `./models/vit-16/`. The directory structure should look like:

```
models/
   |–– vit-16/ 
      |–– zeroshot.pt 
      |–– checkpoint_10.pt
```
## Run baselines

- Compare the performance using CLIP ViT-B/16 of (1) zero-shot, (2) fine-tuned (E2E-FT), (3) WSE (WiSE-FT), and (4) OSE.

```
python src/baselines.py --load=models/vit-16/zeroshot.pt,models/vit-16/checkpoint_10.pt
```

## Run our VRF

- Step 1: Identification and Step 2: Distance Calculation

```
python src/knn_distance.py --load=models/vit-16/zeroshot.pt,models/vit-16/checkpoint_10.pt
```

- Step 3: Sample-Wise Ensembling
```
python src/main.py --load=models/vit-16/zeroshot.pt,models/vit-16/checkpoint_10.pt
```






