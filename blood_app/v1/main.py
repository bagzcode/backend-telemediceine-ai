from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles


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


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

