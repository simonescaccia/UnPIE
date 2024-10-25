# UnPIE
Unsupervised Pedestrian Intention Estimation with Graph Neural Networks

![graph_representation](/images/Figure_4.png)

UnPIE network:
![unpie_network](/images/UnPIE.jpg)

## Credits
Some modules are taken and modified from the following repositories:
- [Pedestrian Intention Estimation (PIE) dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/).
- [Pedestrian intention and trajectory estimation (PIEPredict) model](https://github.com/aras62/PIEPredict).
- [Unsupervised Learning from Video with Deep Neural Embeddings (VIE) model](https://github.com/neuroailab/VIE).

## Setup

### Conda environment

[Miniconda](https://docs.anaconda.com/miniconda/) Linux installation:
  
  ```bash
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
  ```

Check `$PATH`:
  ```bash
  echo $PATH
  ```

Add `export PATH="/home/username/miniconda3/bin:$PATH"` at the end of `~/.profile` file if the directory is not present (substituting username):
  
  ```bash
  echo "export PATH=\"/home/username/miniconda3/bin:\$PATH\"" >> ~/.profile
  ```

Create environment:
  ```bash
  conda env create -f environment_tf2.yaml
  ```

Activate environment:
  ```bash
  conda activate unpie-tf2-env
  ```

Deactivate environment:
  ```bash
  conda deactivate
  ```

### UnPIE setup

Create `config.yml`:

  ```yaml
  PIE_PATH: 'path\to\PIE_dataset'
  PIE_RAW_PATH: 'path\to\PIE_clips'
  PRETRAINED_MODEL_PATH: 'path\to\pretrained\model'
  IS_GPU: False or True
  SETS_TO_EXTRACT: null or ['set01', 'set02', ...] # null for extracting all the sets
  ```

Download annotations and video clips from the [PIE webpage](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) and place them in the `PIE_dataset` directory. 
Annotation zip files should be copied to the main dataset folder and unzipped. There are three types of annotations in the PIE dataset: spatial annotations with text labels, object attributes, ego-vehicle information.

The folder structure should look like this:

```
PIE_dataset
    annotations
        set01
        set02
        ...
    PIE_clips
        set01
        set02
        ...

```

## Preprocessing
Run the following command to extract and save all the image features needed by the GNN without saving each frame:

  ```bash
  python extract_features.py
  ```
![feature_extraction](/images/Feature%20extraction.png)

## Training and testing

  Training:
  ```bash
  sh  run_training.sh
  ```

  Testing:
  ```bash
  sh run_testing.sh
  ```

