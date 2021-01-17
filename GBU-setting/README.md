# CKL-TR（Results evaluated on GBU setting）

## Environments

python 3.7

pytorch-gpu 1.4.0+cuda101

## Data

Data: You can download the GUB-setting datasets from following link：

<https://drive.google.com/file/d/1SF28qqrAZ3dRuzNRIP9JzrKJsLmDaTJh/view?usp=sharing>

Put the uncompressed data to the folder "data", like this:

CKL-TR/GBU-setting

- data
  - AWA2
    - att_splits.mat
    - res101.mat
  
  AWA2.csv
  

## Reproduce results

AwA2：

```python
# run AwA2
python CKL-TR.py --dataset AWA2 --preprocessing --batch_size 512 --attSize 85
```

AwA1：

```python
# run AwA1
python CKL-TR.py --dataset AWA1 --preprocessing --nepoch 2000 --batch_size 512 --attSize 85
```

CUB：

```python
# run CUB
python CKL-TR.py --dataset CUB --preprocessing --batch_size 512 --attSize 312
```

aPY：

```python
# run aPY
python CKL-TR.py --dataset APY --preprocessing --batch_size 512 --attSize 64 --nepoch 5000
```

