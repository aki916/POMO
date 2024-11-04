import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 

import os
import shutil
import time
import numpy as np

from source.utilities import Get_Logger

from HYPER_PARAMS import *
from TORCH_OBJECTS import *


import source.MODEL__Actor.grouped_actors as A_Module
import source.TRAIN_N_EVAL.Train_Grouped_Actors as T_Module
import source.TRAIN_N_EVAL.Evaluate__Grouped_Actors as E_Module

def main():
    SAVE_FOLDER_NAME = "TRAIN_00"
    print(SAVE_FOLDER_NAME)

    # Make Log File
    logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(result_folder_path) 
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    # Objects to Use
    actor = A_Module.ACTOR().to(device)
    actor.optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY)
    actor.lr_stepper = lr_scheduler.StepLR(actor.optimizer, step_size=LR_DECAY_EPOCH, gamma=LR_DECAY_GAMMA)

    # GO
    timer_start = time.time()
    for epoch in range(1, TOTAL_EPOCH+1):
        
        log_package = {
            'epoch': epoch,
            'timer_start': timer_start,
            'logger': logger        
        }


        #  TRAIN
        #######################################################
        T_Module.TRAIN(actor, **log_package)
        

        #  EVAL
        #######################################################
        E_Module.EVAL(actor, **log_package)
        breakpoint()


        #  Check Point
        #######################################################
        checkpoint_epochs = np.arange(1, TOTAL_EPOCH+1, 10)
        if epoch in checkpoint_epochs:
            checkpoint_folder_path = '{}/CheckPoint_ep{:05d}'.format(result_folder_path, epoch)
            os.mkdir(checkpoint_folder_path)
            
            model_save_path = '{}/ACTOR_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(actor.state_dict(), model_save_path)
            optimizer_save_path = '{}/OPTIM_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(actor.optimizer.state_dict(), optimizer_save_path)
            lr_stepper_save_path = '{}/LRSTEP_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(actor.lr_stepper.state_dict(), lr_stepper_save_path)




if __name__ == "__main__":
    main()