### onnxruntime gpu cpp dummy demo

winxos 20240123

#### this is the simplest way to use onnxruntime gpu in cpp

## 1. env prepare

#### !!! you need to install the right version of CUDA and CUDNN

you can found from nvidia site

## 2. create cpp console proj in vs

## 3. install onnxruntime-gpu

you can install package using vcpkg, or nuget

nuget is more convenient. only need onnxruntime-gpu.

## 4. clone this repo

## 5. replace const name in the code

includes:

* model path
* input node name
* output node name
* image path

## 6. have fun.