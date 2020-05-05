# CKL-TR

## Environments

python 3.7

pytorch-gpu 1.4.0+cuda101

## Data

Data: You can download the dataset CUBird and NABird from following link：

<https://drive.google.com/drive/folders/10Jlv2mCaEeFXFhkNbRFsEVHnPMRaZ9jT>

Put the uncompressed data to the folder "data", like this:

CKL-TR

- data
  - CUB2011
  - NABidr
  
  cub.csv
  
  nab.csv

## Reproduce results

CUBird SCS mode && SCE mode

```
# run CUBird with SCS mode
python train_CUB.py --splitmode easy --bird_lambda 0.6 --genus_lambda 0.2 --family_lambda 0.2

# run CUBird with SCE mode
python train_CUB.py --splitmode hard --bird_lambda 0.2 --genus_lambda 0.3 --family_lambda 0.5
```

NABird SCS mode && SCE mode

```
# run NABird with SCS mode
python train_NAB.py --splitmode easy --bird_lambda 0.6 --genus_lambda 0.3 --family_lambda 0.1

# run NABird with SCE mode
python train_NAB.py --splitmode hard --bird_lambda 0.4 --genus_lambda 0.3 --family_lambda 0.3
```