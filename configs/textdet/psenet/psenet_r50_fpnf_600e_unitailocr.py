_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_adam_step_600e.py',
    '../../_base_/det_models/psenet_r50_fpnf.py',
    '../../_base_/det_datasets/unitailocr.py',
    '../../_base_/det_pipelines/unitail_pipeline.py'
]

model = {{_base_.model_quad}}

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.pse_train_pipeline_unitailocr}}
test_pipeline_icdar2015 = {{_base_.pse_test_pipeline_unitailocr}}

# load_from='exps/fromweb/psenet_r50_fpnf_600e_icdar2015-c6131f0d.pth'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015))

evaluation = dict(interval=10, metric='hmean-iou')
