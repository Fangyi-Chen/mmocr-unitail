_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_sgd_1500e.py',
    '../../_base_/det_models/fcenet_r50_fpn.py',
    '../../_base_/det_datasets/unitailocr.py',
    '../../_base_/det_pipelines/unitail_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_unitailocr = {{_base_.fce_train_pipeline_unitailocr}}
test_pipeline_unitailocr = {{_base_.fce_test_pipeline_unitailocr}}

load_from = None

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_unitailocr),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_unitailocr),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_unitailocr))

evaluation = dict(interval=25, metric='hmean-iou')
