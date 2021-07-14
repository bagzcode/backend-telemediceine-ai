from typing import Optional

from fastapi import FastAPI, Header
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles

import base64
from PIL import Image
from io import BytesIO


import os
import sys
sys.path.append('/app/app/covid_next/')

from detect_covid import Classifier
classifier = Classifier("/app/app/covid_next/COVIDNext50_NewData_F1_92.98_step_10800.pth")

class ImageData(BaseModel):
    data: str

    class Config:
        schema_extra = {
                "example": {
                            "data": "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"
                    }
                }


app = FastAPI(
        title="NuMed",
        description="NuMed-CXR: Documentation",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        root_path="/api/v1/cxr-app"
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="/app/app/static"), name="static")

# openapi_url=app.openapi_url,
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/api/v1/cxr-app/openapi.json",
        title=app.title,
        swagger_favicon_url="/static/logo.png",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/api/v1/cxr-app/openapi.json",
        title=app.title + " - ReDoc",
        redoc_favicon_url="/static/logo.png",
    )

@app.get("/", include_in_schema=False)
async def root():
    response = RedirectResponse(url='/api/v1/cxr-app/docs')
    return response


# @app.get("/items/{item_id}")
# def read_item(item_id: int, user_agent: Optional[str] = Header(None)):
#     return {"item_id": item_id, "User-Agent": user_agent}



@app.post("/image/")
async def upload_image(image: ImageData):
    # print(image)
    # img = base64.b64decode(str('iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAApElEQVR42u3RAQ0AAAjDMO5fNCCDkC5z0HTVrisFCBABASIgQAQEiIAAAQJEQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAQECBAgAgJEQIAIyPcGFY7HnV2aPXoAAAAASUVORK5CYII='))  

    # img = base64.b64decode(str(image.data))  
    # img = Image.open(BytesIO(img)) 
    # rotated = img.rotate(45)
    # buffered = BytesIO()
    # rotated.save(buffered, format="PNG")
    # img_str = base64.b64encode(buffered.getvalue())
    # result = {
    #     "image_data": img_str,
    #     "pneumonia" : "60",
    #     "covid" : "20",
    #     "normal" : "20"
    # }

    try:
        res_classifier = classifier.detect(str(image.data))

        result = {
            "status":"success",
            "image_data": res_classifier['img_str'],
            "pneumonia" : res_classifier['result']['pneumonia'],
            "covid" : res_classifier['result']['covid'],
            "normal" : res_classifier['result']['normal']
        }

        return result

    except Exception as e:

        result = {
            "status":"error",
            "image_data": "0",
            "pneumonia" : "0",
            "covid" : "0",
            "normal" : "0",
            "err_message": str(e)
        }

        return result