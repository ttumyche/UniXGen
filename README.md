# Unified Chest X-ray and Radiology Report Generation Model with Multi-view Chest X-rays ([Arxiv](https://arxiv.org/abs/2302.12172))

<img width="1626" alt="nbme_architecture" src="https://user-images.githubusercontent.com/123858584/226160635-ff47d23a-e35f-45ec-aeb0-06a823f50e5d.png">


## Install
~~~
pip install -r requirements.txt
~~~

## Model Weights

####  1. [Chest X-ray Tokenizer](https://drive.google.com/file/d/1CqlKoZQb5FQPUzSk3zanKnFP0qFVSiWu/view?usp=sharing): Download VQGAN and place into the /mimiccxr_vqgan directory
####  2. [UniXGen](https://drive.google.com/file/d/1qS3TFEjpPN-Tjjh6kLE2KElJri77M-Fq/view?usp=share_link): Download the model and place into the /ckpt directory

## Dataset

#### MIMIC-CXR
1. You must be a credential user defined in [PhysioNet](https://physionet.org/settings/credentialing/) to access the data.
2. Download chest X-rays from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and reports from [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)
3. We provide train, valid and test split sets in /metadata directory.


## Train Models

~~~
python unified_main.py
~~~

## Test Models

First, run unified_run.py. \
The generated discrete code sequences are saved as files.
  
~~~
python unified_run.py
~~~

#### For decoding chest X-rays,
Run decode_cxr.py. \
The generated seqeucens for chest X-rays are decoded and saved in the '.jpeg' format.

~~~
python decode_cxr.py
~~~

#### For decoding radiology reports,
Run decode_report.py. \
Save the decoded outputs according to your preference.

~~~
python decode_report.py
~~~
