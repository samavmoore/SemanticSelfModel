from pl_datamodule import VSMDataModule, SemanticVSMDataModule, SemanticVSMDataModule
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_module import VSM, BlendedSDF, DeformationSDF, ClassificationVSM, CorrespondenceVSM
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
import os
import multiprocessing
import threadpoolctl as tpc


architectures = ['siren_og', 'siren_split', 'siren_split_fuse', 'mod_siren', 'mod_relu', 'mod_siren_split']

def stage_0_architecture_trials(architecture, n_trials=5, seed=0, epochs=800):
    n_cpu_cores = multiprocessing.cpu_count()
    n_total_gpus = 8
    n_gpu_used = 1
    num_workers = 8
    limits = max(1, int(n_cpu_cores * n_gpu_used / n_total_gpus / num_workers))

    with tpc.threadpool_limits(limits=limits):

        seed_everything(seed)
        lr = 5e-5
        n_obs = 83200
        devices = 1
        batch_size = 64
        data_dir = './Balanced_Data'

        for i in range(n_trials):
            model =  VSM(lr=lr, epochs=epochs, n_obs=n_obs, devices=devices, batch_size=batch_size, architecture=architecture)
            dm = VSMDataModule(data_dir, batch_size=batch_size, num_workers=8, split_ratio=(1, 0, 0), mode='train_on_surf')
            dm.prepare_data()
            dm.setup()
            checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='newest', every_n_train_steps=40, enable_version_counter=False)
            logger = CSVLogger(save_dir=os.getcwd(), name="stage_0_architecture_logs", version=f'{architecture}_trial_{i}')
            trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[checkpoint_callback], logger=[logger], devices=devices, strategy='auto')
            trainer.fit(model, dm)


def stage_1_trials(architecture, n_trials=5, seed=0, epochs=800):
    n_cpu_cores = multiprocessing.cpu_count()
    n_total_gpus = 8
    n_gpu_used = 1
    num_workers = 8
    limits = max(1, int(n_cpu_cores * n_gpu_used / n_total_gpus / num_workers))

    with tpc.threadpool_limits(limits=limits):

        seed_everything(seed)
        devices = 1
        batch_size = 32
        data_dir = './Balanced_Data'

        for i in range(n_trials):
            #blend_net = VSM.load_from_checkpoint(f'./stage_0_architecture_logs/{architecture}_trial_{0}/checkpoints/newest.ckpt').to('cuda')
            #sdf = VSM.load_from_checkpoint(f'./stage_0_architecture_logs/{architecture}_trial_{0}/checkpoints/newest.ckpt').to('cuda')

            model_path = "/home/sammoore/Downloads/split_siren/checkpoints/newest.ckpt"
            model =  BlendedSDF(lr_blend_net=5e-5, lr_sdf=1e-6, pretrained_blend_net_path=model_path, pretrained_sdf_path=model_path)
            dm = VSMDataModule(data_dir, batch_size=batch_size, num_workers=8, split_ratio=(1, 0, 0), mode='train_off_surf')
            dm.prepare_data()
            dm.setup()
            checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='newest', every_n_train_steps=40, enable_version_counter=False)
            logger = CSVLogger(save_dir=os.getcwd(), name="stage_1_logs", version=f'{architecture}_trial_{i}')
            trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[checkpoint_callback], logger=[logger], devices=devices, strategy='auto')
            trainer.fit(model, dm)

def stage_2_trials(n_trials=5, seed=0, epochs=800):
    n_cpu_cores = multiprocessing.cpu_count()
    n_total_gpus = 8
    n_gpu_used = 1
    num_workers = 8
    limits = max(1, int(n_cpu_cores * n_gpu_used / n_total_gpus / num_workers))

    with tpc.threadpool_limits(limits=limits):

        seed_everything(seed)
        devices = 1
        batch_size = 64
        data_dir = './segmented_data'

        for i in range(n_trials):
            #blend_net = VSM.load_from_checkpoint(f'./stage_0_architecture_logs/{architecture}_trial_{0}/checkpoints/newest.ckpt').to('cuda')
            #sdf = VSM.load_from_checkpoint(f'./stage_0_architecture_logs/{architecture}_trial_{0}/checkpoints/newest.ckpt').to('cuda')

            model_path = "/home/sammoore/Downloads/split_siren/checkpoints/newest.ckpt"
            classification_model_path = "/home/sammoore/Documents/PointCloud-PointForce/classification_logs/trial_0/checkpoints/newest.ckpt"
            model =  DeformationSDF(lr=5e-5, pretrained_sdf_path=model_path, classifier_net_path=classification_model_path)
            dm = SemanticVSMDataModule(data_dir, batch_size=batch_size, num_workers=8, split_ratio=(1, 0, 0), mode='train')
            dm.prepare_data()
            dm.setup()
            checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='newest', every_n_train_steps=40, enable_version_counter=False)
            logger = CSVLogger(save_dir=os.getcwd(), name="stage_2_logs", version=f'trial_{i}')
            trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[checkpoint_callback], logger=[logger], devices=devices, strategy='auto')
            trainer.fit(model, dm)



def classifier_trials(n_trials=5, seed=10, epochs=800):
    n_cpu_cores = multiprocessing.cpu_count()
    n_total_gpus = 8
    n_gpu_used = 1
    num_workers = 8
    limits = max(1, int(n_cpu_cores * n_gpu_used / n_total_gpus / num_workers))

    with tpc.threadpool_limits(limits=limits):

        seed_everything(seed)
        devices = 1
        batch_size = 64
        data_dir = './classification_data'

        for i in range(n_trials):
            model =  ClassificationVSM(lr=5e-5)
            dm = ClassificationVSMDataModule(data_dir, batch_size=batch_size, num_workers=8, split_ratio=(1, 0, 0), mode='train')
            dm.prepare_data()
            dm.setup()
            checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='newest', every_n_train_steps=40, enable_version_counter=False)
            logger = CSVLogger(save_dir=os.getcwd(), name="classification_logs", version=f'trial_{i}')
            trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[checkpoint_callback], logger=[logger], devices=devices, strategy='auto')
            trainer.fit(model, dm)


def correspondence_trials(n_trials=5, seed=10, epochs=800):
    n_cpu_cores = multiprocessing.cpu_count()
    n_total_gpus = 8
    n_gpu_used = 1
    num_workers = 8
    limits = max(1, int(n_cpu_cores * n_gpu_used / n_total_gpus / num_workers))

    with tpc.threadpool_limits(limits=limits):

        seed_everything(seed)
        devices = 1
        batch_size = 64
        data_dir = './segmented_data_new'

        for i in range(n_trials):
            model =  CorrespondenceVSM(lr=5e-5, data_dir=data_dir)
            dm = SemanticVSMDataModule(data_dir, batch_size=batch_size, num_workers=8, split_ratio=(1, 0, 0), mode='train')
            dm.prepare_data()
            dm.setup()
            checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='newest', every_n_train_steps=40, enable_version_counter=False)
            logger = CSVLogger(save_dir=os.getcwd(), name="correspondence_logs", version=f'trial_{i}')
            trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[checkpoint_callback], logger=[logger], devices=devices, strategy='auto')
            trainer.fit(model, dm)



if __name__ == '__main__':
    #arch = architectures[5]
    #stage_0_architecture_trials(arch, seed=5)

    #arch = architectures[1]
    #stage_1_trials(arch)

    #stage_2_trials()
    #classifier_trials()
    correspondence_trials()