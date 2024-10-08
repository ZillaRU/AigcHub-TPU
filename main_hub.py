from fastapi import FastAPI, Request, APIRouter
from api.base_api import InitMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import asyncio
import os
import sys

from fastapi import FastAPI, HTTPException
import importlib

app = FastAPI()

@app.post("/init/{module_name}")
async def init_module(module_name: str):
    try:
        apps_file = "apps.txt"
        repo_dir = "repo"
        module_found = False

        # 读取 apps.txt 文件
        with open(apps_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                name, url = line.strip().split(", ")
                if name == module_name:
                    module_found = True
                    break

        if not module_found:
            raise HTTPException(status_code=404, detail=f"Module {module_name} not found.")

        # 执行 prepare.sh 和 download.sh
        prepare_script = os.path.join(repo_dir, module_name, "prepare.sh")
        download_script = os.path.join(repo_dir, module_name, "download.sh")

        if os.path.exists(prepare_script):
            os.chmod(prepare_script, 0o755)
            process = await asyncio.create_subprocess_shell(f"{prepare_script}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Error running prepare.sh: {stderr.decode()}")

        if os.path.exists(download_script):
            os.chmod(download_script, 0o755)
            asyncio.create_task(asyncio.create_subprocess_shell(f"{download_script}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE))

        os.execv(sys.executable, ['python3'] + sys.argv)

        return {"message": f"Module {module_name} initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 保存当前工作目录
original_dir = os.getcwd()

# 假设所有模块都在一个父目录下
parent_dir = "./api"
sys.path.append(parent_dir)
module_names = ['emotivoice', 'face_gen'] # 'image_gen', 

routers = []

# 动态导入模块
for module_name in module_names:
    module_dir = os.path.join(parent_dir, module_name+'.py')
    if not os.path.exists(module_dir):
        print('No repository / module named', module_name)
    module = importlib.import_module(module_name)
    if hasattr(module, 'router'):
        routers.append(module.router)

# 添加中间件，传入所有需要初始化的 routers
app.add_middleware(InitMiddleware, routers=routers)

# 注册路由
for router in routers:
    app.include_router(router, )


# app.include_router(face_gen.router, tags=['face edit', 'face enhancement', 'face swap'], prefix='/face_gen')
# app.include_router(chattts.router, tags=['chat tts', 'chatting audio generation', 'ChatTTS'], prefix='/chattts')

@app.get("/")
def read_root():
    return {"message": "Hello, enjoy the service of Airbox!"}
