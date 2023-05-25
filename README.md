# Unified Chest X-ray and Radiology Report Generation Model with Multi-view Chest X-rays ([Arxiv](https://arxiv.org/abs/2302.12172))

<img width="1626" alt="nbme_architecture" src="https://user-images.githubusercontent.com/123858584/226160635-ff47d23a-e35f-45ec-aeb0-06a823f50e5d.png">

## Generated Samples

#### Report-to-Chest X-ray
<img width="1626" alt="gen_cxr" src="https://github.com/ttumyche/UniXGen/assets/64394696/16f5fa85-8640-4a93-a2a5-d33c572dcf26">
(a) Based only on the report, the generated PA in the orange dashed box draws a rather small portion of the consolidation in the lingula, as is written in the report.
Based on an additional lateral view, the generated PA in the blue dashed box draws a consolidation that is of more similar size as that of the original PA.

(b) The generated PA conditioned only on the report (orange dashed box) draws relatively small-sized pleural effusion while the report says “large right pleural effusion”.
However, by adding an additional lateral view (blue dashed box), UniXGen can properly generate the PA view with large pleural effusion.

#### Chest X-ray-to-Report
<img width="1626" alt="gen_report" src="https://github.com/ttumyche/UniXGen/assets/64394696/4a66c0e9-dddb-4783-9f48-73ae84710b3b">
(a) Regardless of the number of chest X-rays input, UniXGen can generate accurate radiology reports covering all disease mentioned in the original report.

(b) The generated report only from a single chest X-ray (orange dashed box) cannot fully capture the abnormalities in the given X-ray. With an additional chest X-ray,
UniXGen can generate a more precise report (blue dashed box) containing all disease as described in the original report.

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
