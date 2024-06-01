### Download and pre-process the ScanNet200 dataset
1. Download the [scannet200 dataset](https://kaldir.vc.in.tum.de/scannet_benchmark/documentation)
2. Process the dataset by using the following command.
```
# generate colors,depth,poses

python preprocess_2d_scannet.py \
--scannet_path="PATH/TO/YOUR/SCANS" \
--output_path="PATH/TO/OUTPUT" \
--frame_skip=10
``` 
