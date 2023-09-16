
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
from core.models.vqvae import VQMotionModel
from core.models.motion_regressor import MotionRegressorModel
from ctl.trainer_regressor import RegressorMotionTrainer
from configs.config import cfg, get_cfg_defaults

def main():


    trans_model = MotionRegressorModel(args = cfg.motion_trans ,pad_value=cfg.train.pad_index )
    vqvae_model = VQMotionModel(cfg.vqvae).eval()

    pkg = torch.load(f"./checkpoints/vqvae/mix/vqvae_motion_best_fid.pt", map_location = 'cpu')
    vqvae_model.load_state_dict(pkg["model"])

    trainer = RegressorMotionTrainer(
        trans_model = trans_model,
        vqvae_model = vqvae_model,
        args = cfg.motion_trans,
        training_args = cfg.train,
        dataset_args = cfg.dataset,
        eval_args = cfg.eval_model,
        model_name = cfg.motion_trans_model_name,
        
    ).cuda()


    trainer.train(cfg.train.resume)
    


if __name__ == '__main__':
    
    encodec_cfg_path = "./checkpoints/motionseq/encodec/encodec.yaml"
    librosa_cfg_path = "./checkpoints/motionseq/librosa/librosa.yaml"
    encodec_sine_cfg_path = "./checkpoints/motionseq/encodec_sine/encodec_sine.yaml"


    cfg = get_cfg_defaults()
    print("loading config from:" , encodec_cfg_path)
    cfg.merge_from_file(encodec_cfg_path)
    cfg.freeze()
    print("\n cfg: \n", cfg)
    
 

    
    main()

