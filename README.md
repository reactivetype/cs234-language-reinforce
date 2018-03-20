# Natural Language Directed Reinforcement Learning

## Download Spatial-Reasoning (SPR) data

```
./download_spr_data.sh
```

## Training

### Training with vanilla VIN network on SPR with terminal state as input

```
python train_spr_planning.py
```

### Training with joint representation model with VIN approach on SPR with Imitation Learning

Example kick-off training VIN model on SPR data as propsed in the report

```
python train_spr_global.py --epochs 30  --k 10 --save_dir trained_models/ --actions 4 --l_h 150 --map_type [local|global|joint]
```

### Acknowledgements 

#### Thanks to the following Authors for sharing data and code that served as a basis for our work

* https://github.com/JannerM/spatial-reasoning
* https://github.com/kentsommer/pytorch-value-iteration-networks



