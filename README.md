# OCK: Unsupervised Dynamics Prediction with Object-Centric Kinematics

[Yeon-Ji Song](https://yeonjisong.github.io/), &nbsp; Suhyung Choi, &nbsp; Jaein Kim, &nbsp; Jin-Hwa Kim, &nbsp; Byoung-Tak Zhang 

[ICCV 2025]() | <a href="" target="_blank">arXiv</a>

![image](figures/architecture.png)

## Abstract
Human perception involves discerning complex multi-object scenes into time-static object appearance (\ie, size, shape, color) and time-varying object motion (\ie, location, velocity, acceleration). This innate ability to unconsciously understand the environment is the motivation behind the success of dynamics modeling. Object-centric representations have emerged as a promising tool for dynamics prediction, yet they primarily focus on the objects' appearance, often overlooking other crucial attributes. In this paper, we propose Object-Centric Kinematics (OCK), a framework for dynamics prediction leveraging object-centric representations. Our model utilizes a novel component named object kinematics, which comprises low-level structured states of objects' position, velocity, and acceleration. The object kinematics are obtained via either implicit or explicit approaches, enabling comprehensive spatiotemporal object reasoning, and integrated through various transformer mechanisms, facilitating effective object-centric dynamics modeling. Our model demonstrates superior performance when handling objects and backgrounds in complex scenes characterized by a wide range of object attributes and dynamic movements. Moreover, our model demonstrates generalization capabilities across diverse synthetic environments, highlighting its potential for broad applicability in vision-related tasks.

## Update
The code is coming soon!

## Installation
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup. In our experiments, we use PyTorch 1.10.1 and CUDA 11.3:

```
conda create -n ock python=3.8.8
conda activate ock
pip install -e .
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

The codebase heavily relies on [SlotFormer](https://github.com/pairlab/SlotFormer) and [OCVP](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction).
Please refer to the step-by-step guidance on how to install the [requirements](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction/blob/master/assets/docs/INSTALL.md).


## Dataset Preparation
All datasets should be downloaded or soft-linked to ./data/. All datasets should be downloaded or soft-linked to ./data/. Or you can modify the data_root value in the config files.

**OBJ3D**: This dataset is adopted from [G-SWM](https://github.com/zhixuan-lin/G-SWM#datasets).
Download it manually from the [Google drive](https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view), or use the script provided in that repo.

**MOVi**:
Please download MOVi-{A-E} from the official [website](https://github.com/google-research/kubric/tree/main/challenges/movi).

**WAYMO**: Download Waymo Open Dataset from their official [website](https://waymo.com/open/). You may need to sign in with your Google account to request access.


## Evaluation
The basic experiment pipeline in this project is:

1. Pre-train object-centric slot models `SAVi` on raw videos. After training, the models should be able to decompose the scene into meaningful objects, represented by a set of slots.
2. Apply pretrained object-centric model to extract slots from videos and save them to disk.
3. Train `OCK` over the extracted slots to learn the dynamics of videos.

### Pretrain SAVi
1. Create a new experiment under `/experiments` directory.
```
python src/01_create_experiment.py -d EXP_DIRECTORY --name NAME
optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Directory where the experiment folder will be created
  --name NAME           Name to give to the experiment
```
Modify the experiment parameters located in `experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/experiment_params.json` to adapt to your dataset and training needs.

2. Then, train SAVi given the specified experiments parameters as:
```
python src/02_train_savi.py -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training]

optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the experiment directory
  --checkpoint CHECKPOINT
                        Checkpoint with pretrained parameters to load
  --resume_training     For resuming training
```

### Train OCK 
3. Train an OCK model with the pretrained SAVi model.
Beforehand, create a new predictor folder in the specfiied experiment directory as:
```
python src/01_create_predictor_experiment.py -d EXP_DIRECTORY --name NAME --predictor_name PREDICTOR_NAME

optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Directory where the predictor experimentwill be created
  --name NAME           Name to give to the predictor experiment
  --predictor_name PREDICTOR_NAME
                        Name of the predictor module to use: ['LSTM', 'Transformer', 'OCK']
```
Modify the experiment parameters located in `experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/YOUR_PREDICTOR_NAME/experiment_params.json` to adapt the predictor training parameters to your dataset and training needs.

4. Train your predictor (i.e., OCK) given the specified experiment parameters and a pretrained SAVi model:
```
python 04_train_predictor.py -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training] -m SAVI_MODEL --name_predictor_experiment NAME_PREDICTOR_EXPERIMENT
```


### Evaluate OCK
To evaluate an object-centric video predictor module (i.e. LSTM, Transformer, OCK) run:
```
python 05_evaluate_predictor.py -d EXP_DIRECTORY -m SAVI_MODEL --name_predictor_experiment NAME_PREDICTOR_EXPERIMENT --checkpoint CHECKPOINT [--num_preds NUM_PREDS]

arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the father exp. directory
  -m SAVI_MODEL, --savi_model SAVI_MODEL
                        Name of the SAVi checkpoint to load
  --name_predictor_experiment NAME_PREDICTOR_EXPERIMENT
                        Name to the directory inside the exp_directory corresponding to a predictor experiment.
  --checkpoint CHECKPOINT
                        Checkpoint with predictor pre-trained parameters to load for evaluation
  --num_preds NUM_PREDS
                        Number of rollout frames to predict for (i.e., 10)
```

You can generate figures and animations using the following script:
```
python src/06_generate_figs_pred.py \
      -d experiments/Obj3D/ \
      --savi_model savi_obj3d.pth \
      --name_predictor_experiment Predictor_OCK \
      --checkpoint OCK_obj3d.pth \
      --num_seqs 5 \
      --num_preds 10
```


## Citation
If you find our work useful in your research, please consider to cite our paper!

## Acknowledgement
Special thanks to the following awesome projects!
- [Slot-Attention](https://github.com/google-research/google-research/tree/master/slot_attention)
- [slot_attention.pytorch](https://github.com/untitled-ai/slot_attention)\
- [SAVi](https://github.com/google-research/slot-attention-video/)
- [SlotFormer](https://github.com/pairlab/SlotFormer/)



## Questions
If you have any questions, comments, or suggestions, please reach out to Yeon-Ji Song (yjsong@snu.ac.kr)
