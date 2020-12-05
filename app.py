import fastbook
fastbook.setup_book()

from fastbook import *
from flask import Flask, request, jsonify
#from fastai.basic_train import load_learner
#from fastai.vision.widgets import *
from fastai import *
from fastai.learner import *
from fastai.vision.all import *
from fastai.vision.widgets import *
from fastai.layers import *
from fastai.torch_core import *

import torch
import numpy as np

from PIL import Image
import torchvision.transforms as transforms
pil2tensor = transforms.ToTensor()

from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner('trained_model.pkl', cpu=True)
classes = learn.dls[1].vocab


def pil2tensor(image,dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )


def predict_single(img_file):
    'function to take image and return prediction'
    print(f"{img_file}\n")

    # prediction = learn.predict(img_file)
    # prediction = learn.predict(Image.open(img_file).convert('RGB'))
    prediction = learn.predict(pil2tensor(Image.open(img_file).convert('RGB'),np.float32))


    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()
