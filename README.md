# Keypoint Perception Adjacency

## Build Environment

1. Make sure [conda](https://www.anaconda.com/products/individual) is installed.

2. Create environment from file:

```bash
conda env create -f environment.yml
```


## Prepare Dataset

1. Download [MARS dataset](http://zhang-lab.cecs.anu.edu.au/Project/project_mars.html) and [keypoints](https://drive.google.com/file/d/16M0Y8yCgMgqkSeJtlh6gBoVfbZKvuEWE/view?usp=sharing).

2. Organize the file tree as below:
```
Keypoint-ReID
└── data
    └── mars
        ├── info/
        ├── bbox_train/
        ├── bbox_test/
        ├── bbox_train_keypoints/
        ├── bbox_test_keypoints/
```



## Run

```bash
# training
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --cfg_file <prefix>/cfg.yaml --data.save_dir logs/<version_number>/ --data.sources ['marspose'] --data.targets ['marspose'] --train.max_epoch <epoch_number>

# testing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --cfg_file logs/<version_number>/<time_stamp_and_machine_name>/cfg.yaml --model.resume logs/<version_number>/<time_stamp_and_machine_name>/model/model.pth.tar-<epoch_number> --test.evaluate
```


