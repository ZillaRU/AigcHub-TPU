from pydantic import BaseModel, Field
import base64
import os
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import tempfile
import soundfile as sf
import logging

app_name = "gptsovits"
logging.basicConfig(level=logging.INFO)

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        from repo.gptsovits.web_app import GptSovits_long, gptsovits_dir
        self.gptsovits_long = GptSovits_long(model_path=f"{gptsovits_dir}/models", tokenizer=f"{gptsovits_dir}/g2pw_tokenizer")
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.gptsovits_long

router = AppInitializationRouter(app_name=app_name)

class TTSRequest(BaseModel):
    ref_audio_path: str = Field(..., description="参考音色语音的路径")
    ref_content: str = Field(..., description="参考语音的文本")
    text_content: str = Field(..., description="要转换为语音的文本")

@router.post("/gptsovits_path")
@change_dir(router.dir)
async def gptsovits_api(request: TTSRequest):  
    try:
        # 调用 gptsovits_long 函数
        sr, wav_np = router.gptsovits_long(request.ref_audio_path, request.ref_content, request.text_content)
        # 保存输出音频
        if not os.path.exists("/data/tmpdir/aigchub"):
            os.makedirs("/data/tmpdir/aigchub")
        audio_path = "/data/tmpdir/aigchub/gptsovits_output.wav"
        sf.write(audio_path, wav_np, sr)
        logging.info("语音转换成功")
        # return {"message": "语音转换成功", "audio_path": audio_path}
        return JSONResponse(content=jsonable_encoder(audio_path), media_type="application/json") # 为方便复制，只返回音频地址
    except Exception as e:
        return {"message": "处理过程中出现错误", "error": str(e)}

class TTSBase64Request(BaseModel):
    ref_audio_base64: str = Field(..., description="参考音色语音的base64编码")
    ref_content: str = Field(..., description="参考语音的文本")
    text_content: str = Field(..., description="要转换为语音的文本")

@router.post("/gptsovits_base64")
@change_dir(router.dir)
async def gptsovits_api_base64(request: TTSBase64Request):  
    try:
        # 解码 base64 编码的语音
        ref_audio_data = base64.b64decode(request.ref_audio_base64)
        
        # 使用临时文件处理解码后的数据
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
            tmp_ref.write(ref_audio_data)
            ref_audio_path = tmp_ref.name

        # 调用 gptsovits_long 函数
        sr, wav_np = router.gptsovits_long(ref_audio_path, request.ref_content, request.text_content)
        # 手动删除文件
        os.remove(ref_audio_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, wav_np, sr)
            with open(tmp_out.name, 'rb') as file:
                audio_data = file.read()
            os.remove(tmp_out.name)
        audio_base64 = base64.b64encode(audio_data).decode()
        logging.info("base64语音转换成功")
        # return audio_base64 # 为方便复制，只返回 base64 编码的音频
        return JSONResponse(content=jsonable_encoder(audio_base64), media_type="application/json") # 为方便复制，只返回 base64 编码的音频
    except Exception as e:
        return {"message": "处理过程中出现错误", "error": str(e)}