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

class ImageData(BaseModel):
    data: str

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
async def create_item(image: ImageData):
    # print(image)
    # img = base64.b64decode(str('iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAApElEQVR42u3RAQ0AAAjDMO5fNCCDkC5z0HTVrisFCBABASIgQAQEiIAAAQJEQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAREQIAICBABASIgQAQECBAgAgJEQIAIyPcGFY7HnV2aPXoAAAAASUVORK5CYII='))  
    img = base64.b64decode(str(image.data))  
    img = Image.open(BytesIO(img)) 
    rotated = img.rotate(45)
    buffered = BytesIO()
    rotated.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str