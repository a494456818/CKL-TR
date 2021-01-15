# CKL-TR（Results evaluated on GBU setting）

## Environments

python 3.7

pytorch-gpu 1.4.0+cuda101

## Data

Data: You can download the GUB-setting datasets from following link：

<https://drive.google.com/file/d/1SF28qqrAZ3dRuzNRIP9JzrKJsLmDaTJh/view?usp=sharing>

Put the uncompressed data to the folder "data", like this:

CKL-TR

- data
  - AWA2
  
  AWA2.csv
  

## Reproduce results

AWA2：

```python
# run AWA2
python CKL-TR.py --dataset AWA2 --preprocessing --batch_size 512 --attSize 85 --lr_dec
```
