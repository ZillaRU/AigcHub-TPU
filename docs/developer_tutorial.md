# 贡献指南【如何给 AigcHub 添加新应用】

## AigcHub-TPU项目结构

```
.
├── README.md # 项目readme
├── api # FastAPI接口定义， repo 中的每个 module 与这里的一个文件对应
├── apps.txt # 应用模块列表
├── docs # 文档
├── main_hub.py # 项目启动入口
├── repo # 模块源码文件夹，其中每个文件夹都能作为一个独立的应用仓库
├── requirements.txt # 项目整体依赖
├── scripts # 项目配置、启动脚本
```

## 如何接入新应用
本项目遵循前后端分离的设计原则。本文档主要介绍应用后端如何添加。

### 一、 构建TPU版应用仓库
> 若已有 TPU 版本的应用仓库，请直接跳到本节的5.，随后按照二、将您的应用接入到本项目即可。
> 
> 若应用本身一个 python 文件就能实现，可不单独建立 github 仓库，直接将应用作为 AigcHub 仓库的一部分，参考二、中的 router定义。

对于应用本身源码结构较复杂的情形，选定需要移植的新应用后，需要依次完成以下步骤来创建这个应用的 TPU 版本。

#### 1. 识别算法方案中的各个模型组件

使用PDB/GDB等调试工具，深入分析算法的源码。这一步的目标是明确源码结构和基本逻辑、识别模型组件。各模型组件的基本构造、输入输出、参数量、推理耗时以及预处理和后处理步骤。建议使用 GPT 类工具辅助分析。

#### 2. 获取需要加速的各模型组件的onnx / jit pt模型
    
编写trace脚本或在从原应用仓库中获取trace脚本，将每个可用TPU显著加速推理的模型trace出计算图（即，把torch模型trace为onnx或jit script）。注意验证：输入一致的情况下，onnx模型推理输出与原始torch模型一致。trace脚本应保留，用于后续检查和复现问题。

#### 3. 使用TPU-MLIR工具转换模型
    
TPU-MLIR的基本用法请参考[TPU-MLIR 快速入门](https://tpumlir.org/docs/quick_start/index.html)。

在**正常情况**下只需要手动执行`model_transform.py` 和 `model_deploy.py` 两条命令即可将上一步获取的模型转换为可在算能TPU上执行的模型。这两条命令的用法以及模型精度比对的方法请参考快速入门文档。

若出现了其他情形，请参考：
- 报错显示*模型中包含TPU-MLIR前端/后端不支持的算子*或*模型中有算子用作了未支持的用法*。
    - 首先，尝试降级onnx export时使用的opset版本、降级torch规避该算子的使用、使用其他TPU-MLIR版本。
    - 若未解决，尝试能否通过算子组合的方式替换掉未支持的算子（在python源码相应的 nn.Module的forward()中修改并重新导出）, 或将涉及未支持算子的操作放在模型之外用cpu执行，尽量先转出可用的bmodel。
- 若仍无法转出模型，请在[TPU-MLIR issues](https://github.com/sophgo/tpu-mlir/issues)反馈问题，附上模型信息、trace脚本、转换模型的命令以及完整的报错 log 或截图。

#### 4. 改写原始代码以调用上一步转换出的模型
    
SG2300X可用的 runtime API 包括：

1. python可用的：tpu_perf [[whl download](https://github.com/ZillaRU/AigcHub-TPU/releases/download/v0.1/tpu_perf-1.2.35-py3-none-manylinux2014_aarch64.whl)] [[docs](https://docs.radxa.com/en/sophon/airbox/model-compile/tpu_inference)]、sophon-sail [[github](https://github.com/sophgo/sophon-sail.git)]以及 untool。

2. c / cpp可用的：bmruntime [[C docs](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/runtime.html#c-interface)] [[CPP docs](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/runtime.html#id1)]
将原本加载、调用torch / ONNX模型做推理的代码，参考runtime API文档进行改写。

#### 5. 将基本功能封装为易于模块化的函数
将基本的功能封装为*过程性函数*（如，emotivoice的`tts()`）或*对象的方法*（如，sd-lcm-tpu 中的StableDiffusionPipeline）。
- 封装为过程性函数时，不要将模型等作为全局变量去调用，否则在接入到 AigcHub 时，可能需要较多的代码重构。
    - 如果此前已经做了封装，可以把模型或pipeline作为函数的入参之一。

### 二、将应用作为模块接入到 AigcHub

#### 1. 给应用仓库新增`aigchub`分支
- 在该分支添加环境配置脚本`prepare.sh`和模型下载脚本`download.sh`。建议将模型文件等必要的大文件直接上传到应用github仓库的 release 中。
- 在应用代码目录的根目录下，测试这两个脚本能否正确安装环境和下载模型，确保应用代码本身能正常运行。
- 另外，要注意环境配置不应与 AigcHub 中其他应用冲突。AigcHub 根目录中 requirements.txt已经指定版本安装的包（如 torch、torchaudio、torchvision），不建议另行指定版本，这可能会导致 torch 重新安装影响其他应用。如果有*兼容性问题*，请尝试*在应用模块内部的代码中解决*或[发布issue并附上新应用仓库地址以及完整报错信息](https://github.com/ZillaRU/AigcHub-TPU/issues)。

#### 2. Fork AigcHub仓库并修改
- 在 repo 目录下执行`git clone 应用的github链接`
- 在 api 目录下定义 FastAPI 接口。每一个py文件中定义了一个router，可参考已有的例子。
- FastAPI的更多具体用法请参考[docs](https://fastapi.tiangolo.com/tutorial/)。

此处以[emotivoice 的 api](../api/emotivoice.py)为例讲解。
```python
from pydantic import BaseModel, Field
import base64
from api.base_api import BaseAPIRouter, change_dir, init_helper
import os
from typing import Optional
import sys
import uuid

class AppInitializationRouter(BaseAPIRouter): # 固定写法：继承BaseAPIRouter
    dir = "repo/emotivoice" # 固定写法：dir = repo/模块名称
    @init_helper(dir) # 固定写法：为了避免修改应用仓库中引用关系导致应用本身不能单独使用，init_helper装饰器会临时改变sys.path
    async def init_app(self): # 固定写法：必须实现这个函数去执行加载模型等必要的应用初始化操作，InitMiddleware限制了这个函数不会被重复执行
        # 具体的模型import 和 加载
        from repo.emotivoice.demo_page import get_models
        models, tone_color_converter, g2p, lexicon = get_models()
        self.models = {
            "models": models, 
            "tone_color_converter": tone_color_converter,
            "g2p": g2p, 
            "lexicon": lexicon
        }
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del models, tone_color_converter, g2p, lexicon

# 固定写法
app_name = "emotivoice"
router = AppInitializationRouter(app_name=app_name)

# 定义一个请求体
class TTSRequest(BaseModel):
    text_content: str = Field(..., description="要转换为语音的文本")
    speaker: str = Field('8051', description="说话人ID")
    emotion: Optional[str] = Field('', description="情感提示")

# 定义具体的功能接口
@router.post("/tts")
@change_dir(router.dir) # 固定写法：用于临时改变当前工作目录，避免对应用代码本身做修改
async def tts_api(request: TTSRequest):
    from repo.emotivoice.demo_page import tts
    # 省略tts()调用等业务代码
    return {"audio_base64": audio_base64}
```

- 在`app.txt`添加新应用模块的`模块名称, github链接, 类别(image / audio / text / ...)`

- 测试单个应用能否正常运行：执行`bash script/run.sh 模块名称`，在 swagger UI（浏览器打开`盒子ip:8000/docs`）中检查能否正常发请求并得到结果。

- 如果修改了基础环境中某些包的设置或版本，还需要测试是否与其他应用有环境冲突：执行`bash script/run.sh 模块名称 其他应用模块`，进一步测试。

#### 2. 提交您的修改到自有的AigcHub仓库并提Pull request给本仓库
Pull request应明确列出新增的应用的用途、是否影响基础环境。

## 如果您对现有文档/应用有改进建议，或建议支持某些新应用，欢迎[发布issue](https://github.com/ZillaRU/AigcHub-TPU/issues)。
