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

      ```
      > virtualenv --system-site-packages -p python3.10 ./venv
      > source venv/bin/activate
      ```

Install dependencies:
`pip3 install -r requirements.txt`

Create `config.yml`:
```yaml
PIE_PATH: 'path\to\PIE_dataset'
PIE_RAW_PATH: 'path\to\PIE_clips'
PRETRAINED_MODEL_PATH: 'path\to\pretrained\model'
IS_GPU: False or True
```
