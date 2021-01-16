# Our baseline18

## Environments

python 3.7

pytorch-gpu 1.4.0+cuda101

## Data

Data: You can download the GUB-setting datasets from following link：

<https://drive.google.com/file/d/1SF28qqrAZ3dRuzNRIP9JzrKJsLmDaTJh/view?usp=sharing>

Put the uncompressed data to the folder "data", like this:

CKL-TR/our-Baseline18

- data
  - AWA2
    - att_splits.mat
    - res101.mat
  
  AWA2.csv
  

## Reproduce results

AWA2：

```python
# run AWA2
python train_GBU.py --dataset AWA2 --syn_num 600 --preprocessing --batch_size 512 --attSize 85 --center_weight 10
```

