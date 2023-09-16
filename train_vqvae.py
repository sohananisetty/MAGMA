
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path

from core.models.vqvae import VQMotionModel
from ctl.trainer import VQVAEMotionTrainer
from configs.config import cfg, get_cfg_defaults

def main():


    model = VQMotionModel(cfg.vqvae)
    

    trainer = VQVAEMotionTrainer(
        vqvae_model = model,
        args = cfg.vqvae,
        training_args = cfg.train,
        dataset_args = cfg.dataset,
        eval_args = cfg.eval_model,
        model_name = cfg.vqvae_model_name,
        
    ).cuda()


    trainer.train(cfg.train.resume)
    


if __name__ == '__main__':

   

    cfg = get_cfg_defaults()
    print("loading config from:" , "./checkpoints/vqvae/mix/vqvae_mix.yaml")
    cfg.merge_from_file("./checkpoints/vqvae/mix/vqvae_mix.yaml")
    cfg.freeze()
    print("output_dir: ", cfg.output_dir , cfg.train.output_dir)
    
    
    main()




#accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py




# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml  