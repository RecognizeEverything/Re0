
## 🚩 TODO
- [ ]  Organize the experimental code
- [ ]  Add requirements.txt
- [ ]  add notes



conda create -n re0 python=3.18.8

---

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0

---

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

---

# Clone the Repository


# Install Dependency


install Cropformer

The submodule has already pulled the code：
git submodule add https://github.com/facebookresearch/detectron2.git submodule/detectron2
git submodule add https://github.com/qqlu/Entity.git submodule/Entity
git submodule add https://github.com/openai/CLIP.git submodule/CLIP
git submodule add https://github.com/ScanNet/ScanNet.git submodule/ScanNet

git clone xxx --recursive


---

Installation of Cropformer [Entity/Entityv2/CODE.md at main · qqlu/Entity](https://github.com/qqlu/Entity/blob/main/Entityv2/CODE.md)


pip install -e submodule/detectron2

cp -r submodule/Entity/Entityv2/CropFormer submodule/detectron2/projects
make -C submodule/detectron2/projects/CropFormer/entity_api/PythonAPI
cd submodule/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops/
sh make.sh 

---

put checkpoint in [qqlu1992/Adobe_EntitySeg at main](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x)

pip install -e submodule/CLIP



Cropformer Checkpoint
[Entity/Entityv2/README.md at main · qqlu/Entity](https://github.com/qqlu/Entity/blob/main/Entityv2/README.md#model-zoo)

copy `cropformer.py` to `submodule/detectron2/projects/CropFormer/demo_cropformer/`

cp cropformer.py submodule/detectron2/projects/CropFormer/demo_cropformer/

cropformer
pip install imageio
pip install scikit-image
pip install -U openmim
mim install mmcv
pip install timm
pip install fire
pip install natsort

visiualization
pip install open3d