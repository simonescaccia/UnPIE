<div align="center">

# UnPIE: Unsupervised pedestrian intention estimation through deep neural embeddings and spatio-temporal graph convolutional networks

> [Unsupervised pedestrian intention estimation through deep neural embeddings and spatio-temporal graph convolutional networks](https://link.springer.com/article/10.1007/s10044-025-01483-0), 
> Simone Scaccia, Francesco Pro, Irene Amerini, 2025

![graph_representation](/images/Figure_4.png)

## UnPIE network
![unpie_network](/images/unpie2.png)

## Unsupervised training visualization
Instance Recognition method  |  Local Aggregation method
:---------------------------:|:---------------------------:
![](/images/mem_banks_ir.png)  |  ![](/images/mem_banks_la.png)

</div>

## Training/Testing Dataset

### PIE dataset

Download annotations and video clips from the [PIE webpage](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) and place them in the `PIE_dataset` directory, 

You can run this command to get the videos:

```
wget -r -np -c -nH -R index.html https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/
```

**Note**: download all the sets to run training and cross-validation, or only *set03* to run testing: 

```
wget -r -np -c -nH -R index.html https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/
```


Annotation zip files should be copied to the main dataset folder and unzipped. There are three types of annotations in the PIE dataset: spatial annotations with text labels, object attributes, ego-vehicle information.

You can run this command to get the annotations:

```
cd PIE_dataset
wget -O annotations.zip https://github.com/aras62/PIE/blob/master/annotations/annotations.zip?raw=true
wget -O annotations_attributes.zip https://github.com/aras62/PIE/blob/master/annotations/annotations_attributes.zip?raw=true
wget -O annotations_vehicle.zip https://github.com/aras62/PIE/blob/master/annotations/annotations_vehicle.zip?raw=true
unzip 'annotations*.zip' && rm annotations*.zip
```

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


### PSI dataset

Download PSI dataset from **url**. Follow the instructions:

``` bash
python dataset/psi_extend_intent_annotation.py
```


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

Add `export PATH="/home/username/miniconda3/bin:$PATH"` at the end of `~/.profile` file if the directory is not present (*substituting username*):
  
  ```bash
  echo "export PATH=\"/home/username/miniconda3/bin:\$PATH\"" >> ~/.profile
  ```

Reboot the system. Create environment:
  ```bash
  conda env create -f environment_tf2.yaml
  ```

Init conda:
  ```bash
  conda init
  ```

Activate environment:
  ```bash
  conda activate unpie-tf2-env
  ```


### Docker environment

Build Dockerfile
  ```bash
  docker build -t unpie:v1 .
  ```

Run docker container
  ```bash
  docker run --rm -it -v /local/path/to/PIE_dataset:/PIE_dataset --name unpie-v1-c1 --gpus device= unpie:v1
  ```


## UnPIE setup

Create `config.yml` in the repo root:

  ```yaml
  PIE_PATH: 'path\to\PIE_dataset'
  PIE_RAW_PATH: 'path\to\PIE_clips'
  PRETRAINED_MODEL_PATH: 'path\to\pretrained\model'
  IS_GPU: False or True
  SETS_TO_EXTRACT: null or ['set01', 'set02', ...] # null for extracting all the sets
  ```


## Preprocessing
Run the following command to extract and save all the image features needed by the GNN without saving each frame:

  ```bash
  python extract_features.py
  ```
![feature_extraction](/images/feature_extraction_2_white.png)


## Training and testing

  Training:
  ```bash
  sh  run_training_x.sh
  ```

  Testing:
  ```bash
  sh run_testing_x.sh
  ```

where x can be SUP for supervised learning, IR or IR_LA for unsupervised learning.


## Citation
If you find our work useful in your research, please consider citing our publications:
```bibtex
@Article{Scaccia2025,
    author  = {Scaccia, Simone and Pro, Francesco and Amerini, Irene},
    title   = {Unsupervised pedestrian intention estimation through deep neural embeddings and spatio-temporal graph convolutional networks},
    journal = {Pattern Analysis and Applications},
    year    = {2025},
    month   = {May},
    issn    = {1433-755X},
    doi     = {10.1007/s10044-025-01483-0},
    url     = {https://doi.org/10.1007/s10044-025-01483-0}
}


```

## Credits
Some modules are taken and modified from the following repositories:
- [Pedestrian Intention Estimation (PIE) dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/).
- [Pedestrian intention and trajectory estimation (PIEPredict) model](https://github.com/aras62/PIEPredict).
- [Scene-STGCN](https://github.com/tue-mps/Scene-STGCN)
- [Unsupervised Learning from Video with Deep Neural Embeddings (VIE) model](https://github.com/neuroailab/VIE).

