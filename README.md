# NN_operator_CUDA
This is a tiny example of pytorch NN operator by CUDA.

# Usage

## JIT
Directly run the python code.
```
python pytorch/main.py
```

## Setup
```
python pytorch/setup.py install
python pytorch/main.py --compiler setup
```
Uninstall after use
```
pip uninstall tiny_operator
```

## Cmake
```
mkdir build
cd build
cmake ../pytorch
make
cd ..
python pytorch/main.py --compiler cmake
```
