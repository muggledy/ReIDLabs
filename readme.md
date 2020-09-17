Experiments on Windows about re-ID

# Requirements

- python: 3.6
- tensorflow-gpu: 2.0.0
- keras: 2.3.1
- CUDA: 10.0
- cuDNN: 7.6.4
- Pytorch: 1.2.0

### Environment Setting

更新conda镜像源（上交）：

```
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

或通过`-c`临时指定：`conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ gevent`

更新pip镜像源（清华），资源管理器地址栏输入：`%APPDATA%`，新建pip目录，其中再新建pip.ini，输入内容：

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```

或通过`-i`临时指定：`pip install numpy==1.16.4 -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple/`

从*./environment.yaml*安装虚拟环境：`conda env create -f environment.yaml`

激活虚拟环境：`activate reid`

导出环境配置：`conda env export -n reid > environment.yaml`

# Run

Create virtual environment with */environment.yaml*, then run demos in */src/*

