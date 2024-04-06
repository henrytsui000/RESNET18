# RESNET18
A new version of resnet18

## Requirements
> only needs torch, torchsummary, torchvision
```shell
$pip install -r requirements.txt
```

```shell
$python train.py normal
$python train.py modified
```



## Origin RESNET18
```
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```

## Modified RESNET18
```
================================================================
Total params: 6,998,858
Trainable params: 6,998,858
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.13
Params size (MB): 26.70
Estimated Total Size (MB): 39.84
----------------------------------------------------------------
```