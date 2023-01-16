# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import time

from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi

from app import config as cfg
from app.routes import detection, kie, ocr, recognition

app = FastAPI(title=cfg.PROJECT_NAME, description=cfg.PROJECT_DESCRIPTION, debug=cfg.DEBUG, version=cfg.VERSION)


# Routing
app.include_router(recognition.router, prefix="/recognition", tags=["recognition"])
app.include_router(detection.router, prefix="/detection", tags=["detection"])
app.include_router(ocr.router, prefix="/ocr", tags=["ocr"])
app.include_router(kie.router, prefix="/kie", tags=["kie"])


# Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Docs
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=cfg.PROJECT_NAME,
        version=cfg.VERSION,
        description=cfg.PROJECT_DESCRIPTION,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
