from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO

import sys
sys.path.append('/app/app/')

from data_utils import auto_body_crop, multi_ext_file_iter, IMG_EXTENSIONS
from run_covidnet_ct import create_session, load_graph, load_ckpt

# Model directory, metagraph file name, and checkpoint name
MODEL_DIR = '/app/app/models/COVID-Net_CT-2_L'
META_NAME = 'model.meta'
CKPT_NAME = 'model'

# Tensor names
IMAGE_INPUT_TENSOR = 'Placeholder:0'
TRAINING_PH_TENSOR = 'is_training:0'
FINAL_CONV_TENSOR = 'resnet_model/block_layer4:0'
CLASS_PRED_TENSOR = 'ArgMax:0'
CLASS_PROB_TENSOR = 'softmax_tensor:0'
LOGITS_TENSOR = 'resnet_model/final_dense:0'

# Class names, in order of index
CLASS_NAMES = ('Normal', 'Pneumonia', 'COVID-19')



def load_and_preprocess_simple(image_str, width=512, height=512, autocrop=True):
    """Loads and preprocesses images for inference"""
    images = []
    
    img = base64.b64decode(image_str)  
    img = Image.open(BytesIO(img)) 
    img =img.convert("RGB")

    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Load and crop image
#     image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if autocrop:
        image, _ = auto_body_crop(image)
    image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)

    # Convert to float in range [0, 1] and stack to 3-channel
    image = image.astype(np.float32) / 255.0
    image = np.stack((image, image, image), axis=-1)
    
    images.append(image)
    
    return np.array(images)


def make_gradcam_graph(graph):
    """Adds additional ops to the given graph for Grad-CAM"""
    with graph.as_default():
        # Get required tensors
        final_conv = graph.get_tensor_by_name(FINAL_CONV_TENSOR)
        logits = graph.get_tensor_by_name(LOGITS_TENSOR)
        preds = graph.get_tensor_by_name(CLASS_PRED_TENSOR)

        # Get gradient
        top_class_logits = logits[0, preds[0]]
        grads = tf.gradients(top_class_logits, final_conv)[0]

        # Comute per-channel average gradient
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
    return final_conv, pooled_grads
    

def run_gradcam(final_conv, pooled_grads, sess, image):
    """Creates a Grad-CAM heatmap"""
    with graph.as_default():
        # Run model to compute activations, gradients, predictions, and confidences
        final_conv_out, pooled_grads_out, class_pred, class_prob = sess.run(
            [final_conv, pooled_grads, CLASS_PRED_TENSOR, CLASS_PROB_TENSOR],
            feed_dict={IMAGE_INPUT_TENSOR: image, TRAINING_PH_TENSOR: False})
        final_conv_out = final_conv_out[0]
        class_pred = class_pred[0]
        class_prob = class_prob[0, class_pred]
        
        # Compute heatmap as gradient-weighted mean of activations
        for i in range(pooled_grads_out.shape[0]):
            final_conv_out[..., i] *= pooled_grads_out[i]
        heatmap = np.mean(final_conv_out, axis=-1)

        # Convert to [0, 1] range
        heatmap = np.maximum(heatmap, 0)/np.max(heatmap)
        
        # Resize to image dimensions
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        
    return heatmap, class_pred, class_prob

    
def run_inference(graph, sess, images, batch_size=1):
    """Runs inference on one or more images"""
    # Create feed dict
    feed_dict = {TRAINING_PH_TENSOR: False}

    # Run inference
    with graph.as_default():
        classes, confidences = [], []
        num_batches = int(np.ceil(images.shape[0]/batch_size))
        for i in range(num_batches):
            # Get batch and add it to the feed dict
            feed_dict[IMAGE_INPUT_TENSOR] = images[i*batch_size:(i + 1)*batch_size, ...]

            # Run images through model
            preds, probs = sess.run([CLASS_PRED_TENSOR, CLASS_PROB_TENSOR], feed_dict=feed_dict)

            # Add results to list
            classes.append(preds)
            confidences.append(probs)

    classes = np.concatenate(classes, axis=0)
    confidences = np.concatenate(confidences, axis=0)

    return classes, confidences

# Create full paths
meta_file = os.path.join(MODEL_DIR, META_NAME)
ckpt = os.path.join(MODEL_DIR, CKPT_NAME)

# Load metagraph and create session
graph, sess, saver = load_graph(meta_file)

# Load checkpoint
with graph.as_default():
    load_ckpt(ckpt, sess, saver)
final_conv, pooled_grads = make_gradcam_graph(graph)

class ImageData(BaseModel):
    data: str

    class Config:
        schema_extra = {
                "example": {
                            "data": "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAAAAABWESUoAAAABGdBTUEAALGPC/xhBQAAAAJiS0dEAP+Hj8y/AAAAB3RJTUUH5QcOExQCBmIg4gAAAzpJREFUOMt9k9lPG1cUxu9/Zey593qcjWAzYw94Zjx4wRtewJgt6QNFFWRBUZomL12cLlGVpIuoSITSpiEbyp7QKBjwwtjMeMEG24RIkfIUqVLvDKRKqrbfPN35fvrOuWfmAIPBSMF/ldGgCfynD6FJB4wQ/i8B4D8AhD+sAt43KcpEUQjTtBn9TewBFKRMcJ+tyx32szQ2Y7RHUO8ABJmR0z/N33v6fPFWatQK6XeFTDqAkH3y14xSqShrpWpFWfklZiYhUE8BxIV0bDZbqVeraj5XrNZrivxkcj/S+0AIEPDQ9LM1+fri8tLSal5WFFlW1eXveGTWawCE8CfpbGa66+TjdHp9q7Xdatb/uLuenRN2L0xKcDcz6td2VvwiLTdfbhNtXfHdULMprQ0NgMn02mJAcHG9P5aqDeK/rKVsRwu5hW49AmA0vro+O+jqttuimbraIsTqR53878X0ENS6AO2Wqaw8HRbtnQ7u7NO1zZ3th5/1OblvlcwpSgf20yk5E2ckgRV5znV55dGFPl6SghfruS/1ewC8b0aRo52sp8fhdDDnZgWG6RY4z6VGPoUxmRVAlplyfaSrOxoS+jn+t0eSJPaIkvNys3BeHxXA+PvqzpmRiJsfCDtGy9f4gDvBdfHXmmqKJoA2ybOVnaunj0msIDnn/rwtCG63mxnItip6AgKQmlJrCxNRtsN6eOLV29bHDGvvEX/Y2FS+ofVJ0nA8rxa+ctiZDn7+zeu3Dz0d9tAFtVTMnUDYYrEAW7vvQa6YvzTmjc9sbb9+tTM/efxOXclnbzoPdIhjCRDxc+N3MnJ148VKrVFvbDaa5XKlkMstHGG8scmJw2AgEY/EP736vFgqljbK6xu1UkGWs7c/j4YGh6MhTIGg3xkcjsRHz1y8cuv+48UXz+7O/Xx+ajCa7Bc9ERs0AovL3+sSwkm/f2gomRw5OjY0HPMmBtxir897EJkMZA7Y5vcEfDzv6+sfTCYTsaAkeILuAEeTz63tBSIyW6WQxAfC4VAoHBSFkJe1kD8fozYCtFGYSNso+qCVZRlru2X3SB6jvrxtFGGJ9naF0iK1I4Ztu9ttMJj0F+8J66G6b/gLNjK5QO2S63QAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjEtMDctMTRUMTk6MTk6NTYrMDA6MDAOLCp3AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIxLTA3LTE0VDE5OjE5OjU2KzAwOjAwf3GSywAAACB0RVh0c29mdHdhcmUAaHR0cHM6Ly9pbWFnZW1hZ2ljay5vcme8zx2dAAAAGHRFWHRUaHVtYjo6RG9jdW1lbnQ6OlBhZ2VzADGn/7svAAAAGHRFWHRUaHVtYjo6SW1hZ2U6OkhlaWdodAAxOTJAXXFVAAAAF3RFWHRUaHVtYjo6SW1hZ2U6OldpZHRoADE5MtOsIQgAAAAZdEVYdFRodW1iOjpNaW1ldHlwZQBpbWFnZS9wbmc/slZOAAAAF3RFWHRUaHVtYjo6TVRpbWUAMTYyNjI5MDM5NlnFLnsAAAAPdEVYdFRodW1iOjpTaXplADBCQpSiPuwAAABWdEVYdFRodW1iOjpVUkkAZmlsZTovLy9tbnRsb2cvZmF2aWNvbnMvMjAyMS0wNy0xNC8xMWU2OGQ1YWYwNTU4YzE0YTM0ZmQ4Njc4YjVlMDNjOC5pY28ucG5nZkJkYwAAAABJRU5ErkJggg=="
                    }
                }

app = FastAPI(
        title="NuMed",
        description="NuMed-CT: Documentation",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        root_path="/api/v1/ct-app"
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="/app/app/static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/api/v1/ct-app/openapi.json",
        title=app.title,
        swagger_favicon_url="/static/logo.png",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/api/v1/ct-app/openapi.json",
        title=app.title + " - ReDoc",
        redoc_favicon_url="/static/logo.png",
    )

@app.get("/", include_in_schema=False)
async def root():
    response = RedirectResponse(url='/api/v1/ct-app/docs')
    return response


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/image/")
async def upload_image(image: ImageData):

    try:
        image = load_and_preprocess_simple(str(image.data))

        # Run Grad-CAM
        # heatmap, class_pred, class_prob = run_gradcam(final_conv, pooled_grads, sess, image)
        tmp = run_gradcam(final_conv, pooled_grads, sess, image)

        # print(tmp)
        heatmap = tmp[0]
        class_pred = tmp[1]
        class_prob = tmp[2]

        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)

        superimposed_img = np.uint8(heatmap_resized * 0.4 + 0.6*np.uint8(255 * image[0]))

        im = Image.fromarray(superimposed_img)
        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())

        # result = {
        #     "image_data": img_str,
        #     "diagnosis" : CLASS_NAMES[class_pred],
        #     "confidence" : np.array(class_prob)
        # }

        result = {
            "status":"success",
            "image_data": img_str,
            "diagnosis" : CLASS_NAMES[class_pred],
            "confidence" : float(class_prob)
        }

        return result
    except Exception as e:

        result = {
            "status":"error",
            "image_data": "0",
            "diagnosis" : "0",
            "confidence" : 0,
            "err_message": str(e)
        }

        return result