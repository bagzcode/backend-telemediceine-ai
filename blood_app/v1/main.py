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

import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

def covid_detect(model,X):
    model2 = load_model(model, compile = True)
    probapredict=model2.predict(X)
    classpredict=(probapredict > 0.5).astype('int32')
    return probapredict,classpredict

class Data(BaseModel):
    data: dict
    
    class Config:
        schema_extra = {
                "example": {
                            "data": {
                                'Sex':1,
                                'Age':51.0,
                                'CA': 1.97,
                                'CK': 237.0,
                                'CREA':0.97,
                                'ALP':54.0,
                                'GGT':98.0,
                                'GLU':98.0,
                                'AST':74.0,
                                'ALT':84.0,
                                'LDH':441.0,
                                'CRP':116.5,
                                'K':4.24,
                                'NA':133.7,
                                'UREA':30.0,
                                'WBC':9.2,
                                'RBC':5.21,
                                'HGB':14.9,
                                'HCT':42.7,
                                'MCV':82.0,
                                'MCH':28.6,
                                'MCHC':34.9,
                                'PLT1':337.0,
                                'NE': 71.3350245095,
                                'LY': 19.9802433558,
                                'MO': 7.5754192206,
                                'EO': 0.7748073824,
                                'BA': 0.3310749999,
                                'NET':5.6426761346,
                                'LYT':1.3666621586,
                                'MOT':0.5401349223,
                                'EOT':0.0532303974,
                                'BAT':0.0125285378,
                                'Suspect':1.0
                            }
                    }
                }

app = FastAPI(
        title="NuMed",
        description="NuMed-blood: Documentation",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        root_path="/api/v1/blood-app"
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
        openapi_url="/api/v1/blood-app/openapi.json",
        title=app.title,
        swagger_favicon_url="/static/logo.png",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/api/v1/blood-app/openapi.json",
        title=app.title + " - ReDoc",
        redoc_favicon_url="/static/logo.png",
    )

@app.get("/", include_in_schema=False)
async def root():
    response = RedirectResponse(url='/api/v1/blood-app/docs')
    return response


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/data")
async def insert_data(data: Data):  
    try:
        df = pd.DataFrame(data=data.data,index=[0])
        a = covid_detect('/app/app/covid_detect',df)
        probapredict=np.asarray(a)[0][0][0]
        classpredict=np.asarray(a)[1][0][0]

        if(probapredict<0.5):
            probapredict = 1-probapredict
            classpredict = (classpredict+1)%2

        return {"status":"success","probapredict":probapredict,'classpredict':classpredict}
    except Exception as e:
        return {"status":"error", "probapredict":0,'classpredict':0,"err_message": str(e)}
