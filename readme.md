Experiments on Windows about re-ID

# Requirements

- python: 3.6
- tensorflow-gpu: 2.0.0
- keras: 2.3.1
- CUDA: 10.0
- cuDNN: 7.6.4

# Installation

We need to install external package of LOMO and Salience descriptors

1. `git clone --depth 1 git@github.com:muggledy/lomo-xqda.git --branch code-mirror --single-branch src/code/lomo`

2. `git clone --depth 1 git@github.com:muggledy/person-reidentification-patch-based.git src/code/salience`

# Run

Create virtual environment with */environment.yaml*, then run demos in */src/*