# 创建一个类继承APIRouter，并且在 init 的时候调用一些初始化加载模型的逻辑

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import subprocess
import asyncio
import os
from utils.file_utils import *
from api.repo_url import repo_urls


class BaseAPIRouter(APIRouter):
    def __init__(self, app_name: str):
        super().__init__()
        self.app_name = app_name
        self.models = {}
    
    async def download_models(self):
        original_dir = os.getcwd()
        os.chdir(f"repo/{self.app_name}")
        _path = "./download.sh"
        print("Downloading models...")
        add_executable_permission(_path)
        try:
            await self._run_script(_path)
            print("Models downloaded.")
        except subprocess.CalledProcessError as e:
            # 如果脚本执行失败，返回错误信息
            raise HTTPException(status_code=500, detail=f"Failed to execute {os.getcwd()}/download.sh: {e}")
        os.chdir(original_dir)

    async def init_app(self, sync_src=False):
        original_dir = os.getcwd()

        if not os.path.exists(f"repo/{self.app_name}"):
            os.chdir(f"repo")
            # clone and rename
            subprocess.run(['git', 'clone', repo_urls[self.app_name]["git_url"], self.app_name, "-b aigchub", "--single-branch"], check=True) 
            print(f"New repository are cloned from {repo_urls[self.app_name]["git_url"]}.")
        # 更新代码
        elif sync_src:
            os.chdir(f"repo/{self.app_name}")
            print("Syncing source code...")
            subprocess.run(['git', 'fetch'], check=True)
            subprocess.run(['git', 'checkout', 'aigchub'], check=True)
            subprocess.run(['git', 'pull', 'origin', 'aigchub'], check=True)
            print("Source code synced.")
        
        # 执行 prepare.sh 配置环境
        _path = "./prepare.sh"
        print("Preparing environment...")
        add_executable_permission(_path)
        try:
            await self._run_script(_path)
            print("Environment prepared.")
        except subprocess.CalledProcessError as e:
            # 如果脚本执行失败，返回错误信息
            raise HTTPException(status_code=500, detail=f"Failed to execute {os.getcwd()}/prepare.sh: {e}")
        os.chdir(original_dir)

    async def _run_script(self, script_path):
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, script_path, stderr.decode())
    