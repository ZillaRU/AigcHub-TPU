# 创建一个类继承APIRouter，并且在 init 的时候调用一些初始化加载模型的逻辑

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import asyncio
import os
from starlette.middleware.base import BaseHTTPMiddleware
from abc import ABC, abstractmethod


class InitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, routers):
        super().__init__(app)
        self.routers = routers  # 这是一个列表，包含所有需要初始化的 router 实例
        self.init_locks = {id(router): asyncio.Lock() for router in routers}  # 为每个 router 创建一个锁

    async def dispatch(self, request: Request, call_next):
        # 检查每个 router 是否已初始化
        for router in self.routers:
            if not router.initialized:
                # 获取对应 router 的锁
                lock = self.init_locks[id(router)]
                async with lock:
                    # 双重检查是否已初始化
                    if not router.initialized:
                        try:
                            await router.init_app()  # 执行初始化
                            router.initialized = True
                        except Exception as e:
                            # 如果初始化失败，返回错误响应
                            return JSONResponse(
                                status_code=500,
                                content={"message": f"Initialization failed for {router.app_name}: {str(e)}"}
                            )
        # 所有 router 都已初始化，处理请求
        response = await call_next(request)
        return response


class BaseAPIRouter(APIRouter, ABC):
    def __init__(self, app_name: str):
        super().__init__()
        self.app_name = app_name
        self.dir = f'./repo/{app_name}'
        if not os.path.exists(self.dir):
            print(f"******** ERROR *********\nApplication {app_name} not found. \nPlease check whether the app has been installed by init_app.sh.\n************************")
            raise NotImplementedError
        self.models = {}
        self.initialized = False

    @abstractmethod
    def init_app(self):
        pass

    @abstractmethod
    def destroy_app(self):
        pass

from functools import wraps

def change_dir(new_dir):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ori_dir = os.getcwd()
            os.chdir(new_dir)
            try:
                result = await func(*args, **kwargs)
            finally:
                os.chdir(ori_dir)
            return result
        return wrapper
    return decorator

import sys
def init_helper(new_dir):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ori_dir = os.getcwd()
            sys.path.append(os.path.join(ori_dir, new_dir))
            os.chdir(new_dir)
            try:
                result = await func(*args, **kwargs)
            finally:
                os.chdir(ori_dir)
                sys.path.remove(os.path.join(ori_dir, new_dir))
            return result
        return wrapper
    return decorator