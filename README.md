# DBNet-lite-pytorch


***

#### data format
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


#### train 


```
python3 train.py 
```

#### test

```
python3 inference.py
```

***

## ToDoList
- [x] tranform DB code format from MhLiao/DB
- [] add light backbone
- [] pruned big model by channel clipping
- [] Model distillation




# reference

 1. https://github.com/whai362/PSENet
 2. https://github.com/MhLiao/DB


