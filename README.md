# Temporal segmentation of sign language videos
This repository is forked from the original repo for the following two papers:

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/) and [Gül Varol](https://www.robots.ox.ac.uk/~gul),
*Sign language segmentation with temporal convolutional networks*, ICASSP 2021.  [[arXiv]](https://arxiv.org/abs/2011.12986)

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Neil Fox](https://www.ucl.ac.uk/dcal/people/research-staff/neil-fox), [Gül Varol](https://www.robots.ox.ac.uk/~gul) and [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/),
*Sign Segmentation with Changepoint-Modulated Pseudo-Labelling*, CVPRW 2021. [[arXiv]](https://arxiv.org/abs/2104.13817)

[[Project page]](https://www.robots.ox.ac.uk/~vgg/research/signsegmentation/)

We implemented the ASFormer model based on the following BMVC 2021 paper: [ASFormer: Transformer for Action Segmentation](https://arxiv.org/pdf/2110.08568.pdf) 

The goal of this repository is to experiment with Transformers on the sign segmentation task, if you want to work with the MS-TCN please refer to [github.com/RenzKa/sign-segmentation](https://github.com/RenzKa/sign-segmentation/).

## Contents
* [Setup](#setup)
* [Data and models](#data-and-models)
* [Demo](#demo)
* [Training](#training)
  * [Train ICASSP](#train-icassp)
  * [Train CVPRW](#train-cvprw)
* [Citation](#citation)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Setup

``` bash
# Clone this repository
git clone git@github.com:jeanq1/sign-segmentation.git
cd sign-segmentation/
# Create signseg_env environment
conda env create -f environment.yml
conda activate signseg_env
```

## Data and models
You can download our pretrained models (`models.zip [302MB]`) and data (`data.zip [5.5GB]`) used in the experiments [here](https://drive.google.com/drive/folders/17DaatdfD4GRnLJJ0RX5TcSfHGMxMS0Lm?usp=sharing) or by executing `download/download_*.sh`. The unzipped `data/` and `models/` folders should be located on the root directory of the repository (for using the demo downloading the `models` folder is sufficient).


### Data:
Please cite the original datasets when using the data: [BSL Corpus](https://bslcorpusproject.org/cava/acknowledgements-and-citation/) | [Phoenix14](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
We provide the pre-extracted features and metadata. See [here](data/README.md) for a detailed description of the data files. 
- Features: `data/features/*/*/features.mat`
- Metadata: `data/info/*/info.pkl`

## Training
### Train ICASSP
Training for the MSTCN is the same as in the original repository, if you want to run transformer implementations please run the following files : 
* main-transformer.py for the ASFormer implementation
* main-transformer_torch.py for the Transformer encoder only implementation
* main-transformer_annotated.py for the Annotated Transformer implementation

They all use the same parameters as the original main.py running MS-TCN.

An example command to run these files would be : python main-transformer.py --action train --extract_set train --train_data bslcp --test_data bslcp --num_epochs 30 --bz 8



## Citation
If you use this code and data, please cite the following:

```
@inproceedings{Renz2021signsegmentation_a,
    author       = "Katrin Renz and Nicolaj C. Stache and Samuel Albanie and G{\"u}l Varol",
    title        = "Sign Language Segmentation with Temporal Convolutional Networks",
    booktitle    = "ICASSP",
    year         = "2021",
}
```
```
@inproceedings{Renz2021signsegmentation_b,
    author       = "Katrin Renz and Nicolaj C. Stache and Neil Fox and G{\"u}l Varol and Samuel Albanie",
    title        = "Sign Segmentation with Changepoint-Modulated Pseudo-Labelling",
    booktitle    = "CVPRW",
    year         = "2021",
}
```
```

@article{DBLP:journals/corr/abs-2110-08568,
  author    = {Fangqiu Yi and
               Hongyu Wen and
               Tingting Jiang},
  title     = {ASFormer: Transformer for Action Segmentation},
  journal   = {CoRR},
  volume    = {abs/2110.08568},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.08568},
  eprinttype = {arXiv},
  eprint    = {2110.08568},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-08568.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
The license in this repository only covers the code. For data.zip and models.zip we refer to the terms of conditions of original datasets.


## Acknowledgements
The code builds on the [github.com/yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn) and [github.com/ChinaYi/ASFormer](https://github.com/ChinaYi/ASFormer) repositories. The demo reuses parts from [github.com/gulvarol/bsl1k](https://github.com/gulvarol/bsl1k).  We like to thank C. Camgoz for the help with the BSLCORPUS data preparation.