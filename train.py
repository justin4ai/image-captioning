# Setup cell.
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from CV.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from CV.transformer_layers import *
from CV.captioning_solver_transformer import CaptioningSolverTransformer
from CV.classifiers.transformer import CaptioningTransformer
from CV.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from CV.image_utils import image_from_url

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # Set default size of plots.
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load COCO data from disk into a dictionary.
data = load_coco_data(pca_features=True)

# Print out all the keys and values from the data dictionary.
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))

torch.manual_seed(231)
np.random.seed(231)

data = load_coco_data(pca_features=False)

transformer = CaptioningTransformer(
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          wordvec_dim=256,
          num_heads=2,
          num_layers=2,
          max_length=30
        )



transformer_solver = CaptioningSolverTransformer(transformer, data, idx_to_word=data['idx_to_word'], gpu=True,
           #num_epochs=20,
           num_epochs=1,
           #batch_size=1024,
           batch_size=2048,
           learning_rate=0.0006,
           verbose=True, print_every=10,
         )

transformer_solver.train()

# Plot the training losses.
plt.plot(transformer_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()
plt.savefig("log.png")

import json
import os
from tqdm import tqdm

transformer.to(torch.device("cpu"))

def generate_caption(feature):

    #print(len(feature[0][0])) [[4096 lenght..]]
    sample_caption = transformer.sample(feature, max_length=30)
    print(type(sample_caption))
    sample_caption = decode_captions(sample_caption, data['idx_to_word'])[0]
    return sample_caption



student_id = "2020047029" # FIXME
pred = []
nice_feat = data['nice_feature']
nice_feat = np.expand_dims(nice_feat, axis=1)

for i in tqdm(range(len(nice_feat))):
    #print(nice_feat[i])
    caption = generate_caption(nice_feat[i])
    image_id = i + 1
    pred.append({'image_id' : image_id, 'caption' : caption})

result = {"student_id" : student_id, "prediction" : pred}
json.dump(result, open('pred.json', 'w'))