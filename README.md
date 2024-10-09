# AigcHub-TPU

本项目提供 Airbox（算能SG2300X inside）AIGC能力的一站式体验。
- 连接 Airbox 的方式：请参考 Airbox Wiki。
- 当前可以在Airbox运行的应用如[列表](https://gitee.com/zilla0717/AirboxWiki/blob/master/README.md)所示，部分已经接入到本项目。欢迎参考开发指南将这些应用以及其他实用有趣的应用接入进来。


## 支持应用列表

| 模块名称 | GitHub链接 |  功能描述 |
|--------------|-------------------| ------------------|
| roop_face | https://github.com/ZillaRU/roop_face.git | 人像换脸、人脸修复、人脸增强 |
| sd_lcm_tpu | https://github.com/ZillaRU/SD-lcm-tpu.git | 文生图、图生图、语义超分 |
| emotivoice | https://github.com/ZillaRU/EmotiVoice-TPU.git | 文本转语音（支持情感控制）、音色克隆 |

此处的内容与`app.txt`中一致。app.txt中每行的第一列是该仓库作为模块的名称，第二列是对应的 github 仓库地址。**注意**：
- 这些应用仓库在本项目中作为模块使用，因此**名称可能与原始的仓库不同**（模块名仅能包含字母、数字和下划线，且不能以数字开头）。
- GitHub仓库必须是当前用户有访问权限的（建议直接设置为public）。


## 如何使用 AigcHub 中的已有应用
使用上一节列表中的应用，仅需要按照以下步骤。
### 1. 下载本项目并初始化环境 (初次使用 AigcHub)
```sh
git clone https://github.com/ZillaRU/AigcHub-TPU.git && bash scripts/init_env.sh
```

### 2. 应用初始化 (初次安装某个应用)
\* 如果本项目有更新的版本，建议先执行`git pull`更新本项目代码。

执行下面的命令：
```sh
bash scripts/init_app.sh app.txt中的模块名称
```
其中`app.txt中的模块名称`可以是多个，用空格分开。比如`bash scripts/init_app.sh emotivoice roop_face`。

这一步会从github获取模块的源码、配置环境、下载默认的模型文件。

### 3. 启动指定的后端服务
- 首先在`main_hub.py`中指定需要启动的功能模块。
找到`module_names = [...]`这一行并修改...为所需的模块。
例如，需要使用生图和换脸功能，则应该修改为：
```python
module_names = ['image_gen', 'face_gen']
```
请注意，由于Airbox的 tpu 内存限制，部分应用不能同时启动。

- 执行`bash scripts/run.sh`

- 查看是否正常启动：浏览器访问`盒子ip:8000`
- 查看并测试接口：浏览器访问`盒子ip:8000/docs`，选择对应接口并点击`Try it out`即可在当前选项卡编辑请求并发送，response 将会显示在下方。


## 如何给 AigcHub 添加新应用【贡献指南】
本项目遵循前后端分离的设计原则。因此，本节的添加后端支持、添加前端支持没有先后之分。

### 添加后端支持
#### 一、 构建TPU版应用仓库
> 若已有 TPU 版本的应用，请跳过本节，按照 2. 将您的应用接入到本项目即可。

选定需要移植的新应用后，需要依次完成以下步骤来创建这个应用的 TPU 版本。

##### 1. 识别算法方案中的各个模型组件

使用PDB/GDB等调试工具，深入分析算法的源码。这一步的目标是明确源码结构和基本逻辑、识别模型组件。各模型组件的基本构造、输入输出、参数量、推理耗时以及预处理和后处理步骤。建议使用 GPT 类工具辅助分析。

##### 2. 获取需要加速的各模型组件的onnx / jit pt模型
    
编写trace脚本或在从原应用仓库中获取trace脚本，将每个可用TPU显著加速推理的模型trace出计算图（即，把torch模型trace为onnx或jit script）。注意验证：输入一致的情况下，onnx模型推理输出与原始torch模型一致。trace脚本应保留，用于后续检查和复现问题。

##### 3. 使用TPU-MLIR工具转换模型
    
TPU-MLIR的基本用法请参考[TPU-MLIR 快速入门](https://tpumlir.org/docs/quick_start/index.html)。

在**正常情况**下只需要手动执行`model_transform.py` 和 `model_deploy.py` 两条命令即可将上一步获取的模型转换为可在算能TPU上执行的模型。这两条命令的用法以及模型精度比对的方法请参考快速入门文档。

若出现了其他情形，请参考：
- 报错显示*模型中包含TPU-MLIR前端/后端不支持的算子*或*模型中有算子用作了未支持的用法*。
    - 首先，尝试降级onnx export时使用的opset版本、降级torch规避该算子的使用、使用其他TPU-MLIR版本。
    - 若未解决，尝试能否通过算子组合的方式替换掉未支持的算子（在python源码相应的 nn.Module的forward()中修改并重新导出）, 或将涉及未支持算子的操作放在模型之外用cpu执行，尽量先转出可用的bmodel。
- 若仍无法转出模型，请在[TPU-MLIR issues](https://github.com/sophgo/tpu-mlir/issues)反馈问题，附上模型信息、trace脚本、转换模型的命令以及完整的报错 log 或截图。

##### 4. 改写原始代码以调用上一步转换出的模型
    
SG2300X可用的 runtime API 包括：

1. python可用的：tpu_perf [[whl download](https://github.com/ZillaRU/AigcHub-TPU/releases/download/v0.1/tpu_perf-1.2.35-py3-none-manylinux2014_aarch64.whl)] [[docs](https://docs.radxa.com/en/sophon/airbox/model-compile/tpu_inference)]、sophon-sail [[github](https://github.com/sophgo/sophon-sail.git)]以及 untool。

2. c / cpp可用的：bmruntime [[C docs](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/runtime.html#c-interface)] [[CPP docs](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/runtime.html#id1)]
将原本加载、调用torch / ONNX模型做推理的代码，参考runtime API文档进行改写。

#### 二、将应用作为模块接入到 AigcHub （not done）

##### 1. 给应用仓库新增`aigchub`分支
- 在该分支添加环境配置脚本`prepare.sh`和模型下载脚本`download.sh`。
- 在应用代码目录的根目录下，测试这两个脚本能否正确安装环境和下载模型，确保应用代码本身能正常运行。
- 另外，要注意环境配置不与 AigcHub 中其他应用冲突，不建议指定torch等基础包的版本。

##### 2. Fork AigcHub仓库并修改
- 在`app.txt`添加一行
- 在 repo 目录下执行`git clone xxxxx`
- 在 api 目录下定义 FastAPI 接口
- 修改 main_hub.py，测试能否正常运行和发请求
- 测试是否与其他应用有环境冲突

### 添加前端支持
todo