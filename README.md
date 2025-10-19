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
All datasets should be downloaded or soft-linked to the ./data/ directory.

**OBJ3D**: This dataset is adopted from [G-SWM](https://github.com/zhixuan-lin/G-SWM#datasets).
Download it manually from the [Google drive](https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view), or use the script provided in that repo.

**MOVi**: Please download MOVi-{A-E} from the official [website](https://github.com/google-research/kubric/tree/main/challenges/movi).

**WAYMO**: Download Waymo Open Dataset from their official [website](https://waymo.com/open/). You may need to sign in with your Google account to request for access.

## Training & Evaluation Pipeline
The complete pipeline consists of two main training stages followed by an evaluation stage.

### Stage 1: Pre-train SAVi
First, we pre-train the SAVi model to learn object-centric slot representations from raw videos.
1. Create Experiment: Generate a new experiment directory and configuration file:
```
python src/create_experiment.py -d EXP_DIRECTORY --name NAME
```
2. Configure Parameters: Modify the experiment parameters located in experiments/EXP_DIR/EXP_NAME/experiment_params.json to adapt to your dataset and training needs.
3. Run Training: Start the SAVi training:
```
python src/train_savi.py -d EXP_DIRECTORY [--checkpoint CHECKPOINT]
```

### Stage 2: Train OCK
Once SAVi is pre-trained, you can train the OCK predictor model, which learns the dynamics over the extracted slots.
1. Create Predictor Experiment: Create a new predictor folder within your experiment directory:
```
python src/create_ock_environment.py -d EXP_DIRECTORY --name NAME --baseline_name BASELINE_NAME
```
2. Configure Parameters: Modify the predictor's parameters located in experiments/EXP_DIR/EXP_NAME/BASELINE_NAME/experiment_params.json to suit your training needs.
3. Run Training: Train OCK, providing the path to your pre-trained SAVi model:
```
python src/train_ock.py -d EXP_DIRECTORY -m SAVI_MODEL --baseline_name NAME_PREDICTOR_EXPERIMENT [--checkpoint CHECKPOINT]
```
SAVI_MODEL should be the path to your checkpoint from Stage 1 (e.g., experiments/EXP_DIR/EXP_NAME/checkpoints/model_best.pth).


### Stage 3: Evaluate OCK
Finally, evaluate your trained OCK model on the video prediction task.
1. Run Evaluation Script: Execute test_ock.py, specifying your dataset's config file and the path to your trained OCK weights from Stage 2.
```
python src/test_ock.py \
    --params ock/video_prediction/configs/ock_{dataset}_params.py \
    --weight /path/to/your/trained/OCK_WEIGHT.pth
```
Here, replace {dataset} with obj3d, movi, etc. The weight file will be in your predictor experiment directory (e.g., experiments/YOUR_EXP_DIR/.../YOUR_PREDICTOR_NAME/checkpoints/model_best.pth).
2. Review Metrics: The script will compute and print all evaluation metrics to the console.
3. Generate Visualizations: To save output videos for visualization (to vis/obj3d/$PARAMS/), add the --save_num argument. This will save the specified number of videos.

## TODO
- [] Training script
- [ ] Evaluation script
- [ ] Pretrained weights

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

