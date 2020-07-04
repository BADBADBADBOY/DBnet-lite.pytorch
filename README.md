[English](README_en.md) | 简体中文

# DBNet-lite-pytorch

## 环境配置

```
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


## 在icdar2015的测试结果

|Method| head|extra data|precision(%)| recall(%)  |   hmean(%)|model_file|
| - | - | - | - | - | - |- |
| Resnet18|FPN|no|86.11|   76.45|  80.99|[baiduyun](https://pan.baidu.com/s/1wmbGMoluWlZ97LCqOnwjOg) (extract code: p0bk)|
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


