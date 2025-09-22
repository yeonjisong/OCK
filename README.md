# OCK: Unsupervised Dynamics Prediction with Object-Centric Kinematics

[Yeon-Ji Song](https://yeonjisong.github.io/), &nbsp; Suhyung Choi, &nbsp; Jaein Kim, &nbsp; Jin-Hwa Kim, &nbsp; Byoung-Tak Zhang 

[ICCV 2025](https://iccv.thecvf.com/) | <a href="https://arxiv.org/abs/2404.18423" target="_blank">arXiv</a>

![image](figures/architecture.png)

## Abstract
Human perception involves decomposing complex multiobject scenes into time-static object appearance (i.e., size, shape, color) and time-varying object motion (i.e., position, velocity, acceleration). For machines to achieve human-like intelligence in real-world interactions, understanding these physical properties of objects is essential, forming the foundation for dynamic video prediction. While recent advancements in object-centric transformers have demonstrated potential in video prediction, they primarily focus on object appearance, often overlooking motion dynamics, which is crucial for modeling dynamic interactions and maintaining temporal consistency in complex environments. To address these limitations, we propose OCK, a dynamic video prediction model leveraging object-centric kinematics and object slots. We introduce a novel component named Object Kinematics that comprises explicit object motions, serving as an additional attribute beyond conventional appearance features to model dynamic scenes. The Object Kinematics are integrated into various OCK mechanisms, enabling spatiotemporal prediction of complex object interactions over long video sequences. Our model demonstrates superior performance in handling complex scenes with intricate object attributes and motions, highlighting its potential for applicability in vision-related dynamics learning tasks.


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
All datasets should be downloaded or soft-linked to ./data/.

**OBJ3D**: This dataset is adopted from [G-SWM](https://github.com/zhixuan-lin/G-SWM#datasets).
Download it manually from the [Google drive](https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view), or use the script provided in that repo.

**MOVi**: Please download MOVi-{A-E} from the official [website](https://github.com/google-research/kubric/tree/main/challenges/movi).

**WAYMO**: Download Waymo Open Dataset from their official [website](https://waymo.com/open/). You may need to sign in with your Google account to request for access.

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
To evaluate the video prediction task, please use [test.py](../ock/video_prediction/test.py) and run:
```
python ock/video_prediction/test_vp.py \
    --params ock/video_prediction/configs/ock_{dataset}_params.py \
    --weight $WEIGHT
```
This will compute and print all the metrics.
Besides, it will also save 10 videos for visualization under `vis/obj3d/$PARAMS/`.
If you only want to do visualizations (i.e. not testing the metrics), simply use the `--save_num` args and set it to a positive value.


## Citation
If you find our work useful in your research, please consider citing our paper!


## Acknowledgement
Special thanks to the following awesome projects!
- [Slot-Attention](https://github.com/google-research/google-research/tree/master/slot_attention)
- [slot_attention.pytorch](https://github.com/untitled-ai/slot_attention)
- [SAVi](https://github.com/google-research/slot-attention-video/)
- [SlotFormer](https://github.com/pairlab/SlotFormer/)
- [OCVP](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction)


## Questions
If you have any questions, comments, or suggestions, please reach out to Yeon-Ji Song (yjsong@snu.ac.kr)

