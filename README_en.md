English| [简体中文](README.md)

# DBNet-lite-pytorch

## setup

```
pip install -r requirement.txt
cd models/dcn/
sh make.sh
```

***

## data format for Horizontal or slanted text

follow icdar15 dataset format, x1,y1,x2,y2,x3,y3,x4,y4,label
```

image
│   1.jpg
│   2.jpg   
│		...
label
│   gt_1.txt
│   gt_2.txt
|		...
```
***

## data format for curved text

dataset format, x1,y1,x2,y2,x3,y3,x4,y4 ...xn,yn,label 

The number of N can be inconsistent,The arrangement of points is clockwise or counterclockwise

```
image
│   1.jpg
│   2.jpg   
│		...
label
│   gt_1.txt
│   gt_2.txt
|		...
```

***


## train 

Go to configure config.yaml in the root directory

```
python3 train.py 
```
***


## test

set is_poly = True in config.yaml for curved text , others set is_poly = False

```
python3 inference.py
```
***
## Channel clipping for model compression

### Training section
1. sparse training is performed first. firstly, modify config.yaml to set use_sr to True, and set sr_lr. the larger this setting is, the more pressure it will have.
 pay attention to the fact that it may not converge if it is too large.

```
python3 train.py
```
2. Compression model
Set the parameters of pruned part in config.yaml and run it
```
python3 ./pruned/prune.py
```
3. Re-finetune model
Here, the accuracy will pick up quickly. Generally, you can train 50-100epoch and do your own experiments
```
python3 ./pruned/train_fintune.py
```

### test section

```
python3 ./pruned/prune_inference.py
```

## performance in icdar2015

|Method| head|extra data|prune ratio|model size(M)|precision(%)| recall(%)  |   hmean(%)|model_file|
| - | - | - | - | - | - |- | - |- |
| Resnet18|FPN|no|0|62.6|86.11|   76.45|  80.99|[baiduyun](https://pan.baidu.com/s/1wmbGMoluWlZ97LCqOnwjOg) (extract code: p0bk)|
| Resnet18|DB|no|0.8|20.1|85.55|   76.40|  80.72||

***
## some result
<img src="./show/1.jpg" width=800 height=500 />     
<img src="./show/2.jpg" width=800 height=500 />

***

#### ToDoList
- [x] tranform DB code format from MhLiao/DB
- [x] add some performance
- [ ] add light backbone
- [ ] pruned big model by channel clipping
- [ ] Model distillation




# reference

 1. https://github.com/whai362/PSENet
 2. https://github.com/MhLiao/DB
 3. https://github.com/Jzz24/pytorch_quantization


