# UnPIE
Unsupervised Pedestrian Intention Estimation

## Credits
Some modules are taken and modified from the following repositories:
- [Pedestrian Intention Estimation (PIE) dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/).
- [Pedestrian intention and trajectory estimation (PIEPredict) model](https://github.com/aras62/PIEPredict).
- [Unsupervised Learning from Video with Deep Neural Embeddings (VIE) model](https://github.com/neuroailab/VIE).

## Setup
To install via virtual environment (recommended) follow these steps:

- Linux:

    - Install virtual environment `sudo apt-get install virtualenv`.

    - Create a virtual environment with Python3:

      ```bash
      virtualenv --system-site-packages -p python3.10 ./venv
      source venv/bin/activate
      pip install -U pip
      ```

Install dependencies:
`python3 -m pip install -r requirements.txt`

Create `config.yml`:

  ```yaml
  PIE_PATH: 'path\to\PIE_dataset'
  PIE_RAW_PATH: 'path\to\PIE_clips'
  PRETRAINED_MODEL_PATH: 'path\to\pretrained\model'
  IS_GPU: False or True
  SETS_TO_EXTRACT: None or ['set01', ...]
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
Run the following command to preprocess the image features without extracting all the annotated frames:

  ```bash
  python extract_images.py
  ```
