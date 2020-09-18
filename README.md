[English](README_en.md) | 简体中文

# DBNet-lite-pytorch


## 这个项目之后会在这里更新,我把之前的项目都做了下整合[pytorchOCR](https://github.com/BADBADBADBOY/pytorchOCR)

## 环境配置

```
pip install -r requirement.txt
cd models/dcn/
sh make.sh
```

***

## 水平或倾斜文本格式

照着icdar2015的格式, x1,y1,x2,y2,x3,y3,x4,y4,label,
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

## 弧形文本的格式

数据格式, x1,y1,x2,y2,x3,y3,x4,y4 ...xn,yn,label 

n个点组成，n的个数可以不定

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


## 训练部分 

在根目录的config.yaml里配置train部分的一些参数，例如一些图片位置，如果你的图片和gt文件名字是一样的，你可以设置is_icdar2015=False。
如果你不想做验证，可以直接设置start_val_epoch大于n_epoch，如果设置了做验证，会保存一个hmean最高的最优模型。

```
python3 train.py 
```
***


## 测试部分

测试时，配置config.yaml中test部分，对于弧形文本设置is_poly=True,其它非弧形文本设置为False

```
python3 inference.py
```
***
## 模型压缩之通道剪裁

### 训练部分
1. 先进行稀疏训练，首先修改config.yaml将use_sr 设置为True，并设定sr_lr，这个设置越大压的越多，注意设置太大有可能不收敛.

```
python3 train.py 
```
2. 压缩模型
设置好config.yaml中pruned部分参数，运行
```
python3 ./pruned/prune.py
```
3. 重新finetune模型
这里精度会很快回升，一般可以训练50-100epoch，具体自己做实验
```
python3 ./pruned/train_fintune.py
```

### 测试部分

```
python3 ./pruned/prune_inference.py
```

***


## 在icdar2015的测试结果

|Method| head|extra data|prune ratio|model size(M)|precision(%)| recall(%)  |   hmean(%)|model_file|
| - | - | - | - | - | - |- | - |- |
| Resnet18|FPN|no|0|62.6|86.11|   76.45|  80.99|[baiduyun](https://pan.baidu.com/s/1wmbGMoluWlZ97LCqOnwjOg) (extract code: p0bk)|
| Resnet18|DB|no|0.8|20.1|85.55|   76.40|  80.72||
***
## 在icdar2015的测试结果图
<img src="./show/1.jpg" width=800 height=500 />     
<img src="./show/2.jpg" width=800 height=500 />

***

#### 该项目会做
- [x] 转换作者的代码便于阅读和调试
- [x] 展示一些训练结果
- [ ] 加入轻量化的backbone压缩模型
- [ ] 通过通道剪裁压缩DB模型，精度基本不变
- [ ] 通过知识蒸馏进一步提升压缩后模型效果




# 参考

 1. https://github.com/whai362/PSENet
 2. https://github.com/MhLiao/DB
 3. https://github.com/Jzz24/pytorch_quantization


