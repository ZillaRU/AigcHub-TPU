from fastapi import FastAPI, Request, APIRouter

from api import face_gen

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import asyncio


# async def run_precheck_once(router: APIRouter):
#     if router.models is None:
#         router.init_models()

# async def precheck_middleware_factory(router: APIRouter):
#     async def precheck_middleware(request: Request, call_next):
#         await run_precheck_once(router)
#         response = await call_next(request)
#         return response
#     return precheck_middleware

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await face_gen.router.init_app(sync_src=True, download_model=True)

def add_func_set(router:APIRouter, tags:list, prefix:str):
    # router.add_middleware(BaseHTTPMiddleware, dispatch=await precheck_middleware_factory(prefix))
    app.include_router(router, tags=tags, prefix=prefix)
    # precheck_done[prefix] = False

# add_func_set(common_utils.router, tags=["common utils",], prefix="")
add_func_set(face_gen.router, tags=['face edit', 'face enhancement', 'face swap'], prefix='/face_gen')

@app.get("/")
def read_root():
    return {"Hello": "World"}
