dataset_type = 'IcdarDataset'
data_root = '/home/fangyi/data/unitail/unitail-ocr'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/unitailocr_training.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/unitailocr_test.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train_list = [train]

test_list = [test]
