Experiments on Windows about re-ID

# Requirements

- python: 3.6
- tensorflow-gpu: 2.0.0
- keras: 2.3.1
- CUDA: 10.0
- cuDNN: 7.6.4
- Pytorch: 1.2.0


# Run

Create virtual environment with */environment.yaml*, then run demos in */src/*

idea1: 无监督场景，对相机风格进行迁移，具体的做法是迁移相机B的风格到给定的相机A下的行人图像上，从而得到相机A对应的一组生成图像，就可以得到相同ID的图像对，问题转为有监督场景，再结合现有的一些无监督处理方案（可以省略，主要的风格迁移才是核心关键）

idea2: 给定一组摄像头数据，利用风格迁移产生其他摄像头风格图片，以此扩充训练集，训练的时候，由于增加了很多不同摄像头风格，因此网络会挖掘与摄像头无关的深层特征，这样训练得到的特征提取器可以直接用于无监督场景

（这两个idea好像都有人做了）