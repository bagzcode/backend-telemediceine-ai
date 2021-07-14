import torch
from PIL import Image

from model.architecture import COVIDNext50Hooked
from data.transforms import val_transforms

# import matplotlib.pyplot as plt
import numpy as np

from model.layers import Trainable, ConvBn2d

import config

import cv2

from iteround import saferound

import base64
from io import BytesIO

class Classifier():

    def __init__(self,ckpt_pth):

        self.rev_mapping = {idx: name for name, idx in config.mapping.items()}
        weights = torch.load(ckpt_pth,map_location=torch.device('cpu'))['state_dict']

        model = COVIDNext50Hooked(n_classes=len(self.rev_mapping))
        model.load_state_dict(weights)
        model.eval();
        self.model = model;

        self.transforms = val_transforms(width=config.width, height=config.height)

    # def detect(self, img_pth, output_pth):
    def detect(self, img_str):

        img = base64.b64decode(img_str)  
        img = Image.open(BytesIO(img)) 

        img = img.convert("RGB")

        # img = Image.open(img_pth).convert("RGB")
        img_tensor = self.transforms(img).unsqueeze(0)

        # detect
        logits = self.model(img_tensor)
        cat_id = int(torch.argmax(logits))

        prob = self.model.probability(logits)

        # GRAD-CAM
        # get the gradient of the output with respect to the parameters of the model
        logits[cat_id].backward()

        # pull the gradients out of the model
        gradients = self.model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.model.get_activations(img_tensor).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # img = cv2.imread(img_pth)
        img = np.array(img) 
        heatmap_resized = cv2.resize(heatmap.detach().numpy(), (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = np.uint8(heatmap_resized * 0.2 + 0.8*img)

        # cv2.imwrite(output_pth, superimposed_img)
        # _, bufferval = cv2.imencode('.jpg', image)
        # bytecode = base64.b64encode(bufferval)

        im = Image.fromarray(superimposed_img)
        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
    
        probability = prob.detach().numpy()
        
#         import pdb; pdb.set_trace();
        prob_percentage = saferound((probability*100).tolist(), places=3)
        # result = {self.rev_mapping[idx]:str(el) for idx, el in enumerate(prob_percentage)}
        result = {'normal':str(prob_percentage[0]), 'pneumonia':str(prob_percentage[1]), 'covid':str(prob_percentage[2])}
        
        return {'result':result, 'img_str':img_str}