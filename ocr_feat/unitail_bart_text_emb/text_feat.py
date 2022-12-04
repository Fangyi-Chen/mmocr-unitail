import os
import pickle
import numpy as np
from tqdm import tqdm

root  = '/media/Brand/haoc/unitail/unitail_ocrtext2hao/ocrfeat_aug_2hao'
train = os.path.join(root, 'train')
test  = os.path.join(root, 'test')
val = os.path.join(root, 'val')
train_list = []
test_list  = []
val_list = []


# record path to train and test data, generate target list
for folder in os.listdir(train):
    imgs = os.listdir(os.path.join(train, folder))
    for i, img in enumerate(imgs):
        image_path = os.path.join(train, folder, img)
        train_list.append(image_path)

for folder in os.listdir(test):
    imgs = os.listdir(os.path.join(test, folder))
    for i, img in enumerate(imgs):
        image_path = os.path.join(test, folder, img)
        test_list.append(image_path)

for folder in os.listdir(val):
    imgs = os.listdir(os.path.join(val, folder))
    for i, img in enumerate(imgs):
        image_path = os.path.join(val, folder, img)
        val_list.append(image_path)

print('{} feats in training'.format(len(train_list)))
print('{} feats in testing'.format(len(test_list)))
print('{} feats in validation'.format(len(val_list)))




from transformers import BartTokenizer, BartModel

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large')


# process train
for file_list in [train_list, test_list, val_list]:
    for file_path in tqdm(file_list):
        file_name = file_path.split('/')[-1]
        save_path = '/'.join(file_path.split('/')[:-1])
        save_path = save_path.replace('unitail_ocrtext2hao', 'unitail_bart_text_emb')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        text = ' '.join(data['texts'])
        char_scores = data['char_scores']
        text_scores = [np.mean(item) for item in char_scores]
        bboxes = data['bboxes']

        if len(bboxes) == 0:
            data['text_bart_embs'] = np.empty(shape=(1024,))
        else:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            text_embedding = outputs.last_hidden_state[0].mean(dim=0).detach().numpy()
            data['text_bart_embs'] = text_embedding
        
        with open(os.path.join(save_path, file_name), 'wb') as f:
            pickle.dump(data, f)


