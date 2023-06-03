_base_ = [
    '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/mmsegmentation/configs/_base_/models/pspnet_r50-d8.py', 
    '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/mmsegmentation/configs/_base_/datasets/potsdam.py',
    '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/mmsegmentation/configs/_base_/default_runtime.py', 
    '/pfs/work7/workspace/scratch/tu_zxmav84-ds_project/data/mmsegmentation/configs/_base_/schedules/schedule_20k.py'
] # base config file which we build new config file on.
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)