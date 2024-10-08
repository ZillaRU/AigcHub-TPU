# 创建一个类继承APIRouter，并且在 init 的时候调用一些初始化加载模型的逻辑

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import subprocess
import asyncio
import os
from utils.file_utils import *
from api.repo_url import repo_urls
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
        self.models = {}
        self.initialized = False

    @abstractmethod
    def init_app(self):
        pass

    @abstractmethod
    def destroy_app(self):
        pass
    # async def download_models(self):
    #     original_dir = os.getcwd()
    #     os.chdir(f"repo/{self.app_name}")
    #     _path = "./download.sh"
    #     print("Downloading models...")
    #     add_executable_permission(_path)
    #     try:
    #         await self._run_script(_path)
    #         print("Models downloaded.")
    #     except subprocess.CalledProcessError as e:
    #         # 如果脚本执行失败，返回错误信息
    #         raise HTTPException(status_code=500, detail=f"Failed to execute {os.getcwd()}/download.sh: {e}")
    #     os.chdir(original_dir)

    # async def init_app(self, sync_src=False):
    #     original_dir = os.getcwd()

    #     if not os.path.exists(f"repo/{self.app_name}"):
    #         os.chdir(f"repo")
    #         # clone and rename
    #         subprocess.run(['git', 'clone', repo_urls[self.app_name]["git_url"], self.app_name, "-b aigchub", "--single-branch"], check=True) 
    #         print(f"{self.app_name} is cloned from {repo_urls[self.app_name]['git_url']}.")
    #     # 更新代码
    #     elif sync_src:
    #         os.chdir(f"repo/{self.app_name}")
    #         print("Syncing source code...")
    #         subprocess.run(['git', 'fetch'], check=True)
    #         subprocess.run(['git', 'checkout', 'aigchub'], check=True)
    #         subprocess.run(['git', 'pull', 'origin', 'aigchub'], check=True)
    #         print("Source code synced.")
        
    #     # 执行 prepare.sh 配置环境
    #     _path = "./prepare.sh"
    #     print("Preparing environment...")
    #     add_executable_permission(_path)
    #     try:
    #         await self._run_script(_path)
    #         print("Environment prepared.")
    #     except subprocess.CalledProcessError as e:
    #         # 如果脚本执行失败，返回错误信息
    #         raise HTTPException(status_code=500, detail=f"Failed to execute {os.getcwd()}/prepare.sh: {e}")
    #     os.chdir(original_dir)
    #     self.initialized = True

    # async def _run_script(self, script_path):
    #     proc = await asyncio.create_subprocess_exec(
    #         "bash", "-c", script_path,
    #         stdout=asyncio.subprocess.PIPE,
    #         stderr=asyncio.subprocess.PIPE)
    #     stdout, stderr = await proc.communicate()
    #     if proc.returncode != 0:
    #         raise subprocess.CalledProcessError(proc.returncode, script_path, stderr.decode())

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