from fastapi import FastAPI
from api.base_api import InitMiddleware
import asyncio
import os
import sys
from fastapi import FastAPI, HTTPException
import importlib

import argparse

parser = argparse.ArgumentParser(description="Run AigcHub API")
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host on which to run the API')
parser.add_argument('--port', type=int, default=8000, help='Port on which to run the API')
parser.add_argument('module_name', type=str, help='App module name to load') ## 一次只加载一个模块
args = parser.parse_args()

app = FastAPI()

tags_metadata = [
    {
        "name": "Init",
        "description": "应用初始化相关操作"
    },
    {
        "name": "Image",
        "description": "图像处理与生成"
    },
    {
        "name": "Audio",
        "description": "音频处理与生成"
    },
    {
        "name": "Text",
        "description": "文本处理与生成"
    },
]

# 保存当前工作目录
original_dir = os.getcwd()

# 假设所有模块都在一个父目录下
parent_dir = "./openai_api"
sys.path.append(parent_dir)
module_names = args.module_names

routers = []

# 动态导入模块
for module_name in module_names:
    module_dir = os.path.join(parent_dir, module_name+'.py')
    if not os.path.exists(module_dir):
        print('No repository / module named', module_name)
        raise ValueError
    module = importlib.import_module(module_name)
    if hasattr(module, 'router'):
        routers.append(module.router)

# 添加中间件，传入所有需要初始化的 routers
app.add_middleware(InitMiddleware, routers=routers)

# 从apps.txt获取应用信息
app_meta_info = {}

apps_file = "apps.txt"
with open(apps_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        name, _, tag = line.strip().split(", ")
        app_meta_info[name] = tag

# 注册路由
for router in routers:
    app.include_router(router, prefix='/'+router.app_name, tags=[app_meta_info[router.app_name]])

@app.get("/")
def read_root():
    return {"message": "Hello, enjoy the services supported by Airbox on http://0.0.0.0:8000/docs !"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
