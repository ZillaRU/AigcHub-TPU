from pydantic import BaseModel, Field
import base64
from api.base_api import BaseAPIRouter, change_dir, init_helper
import os
from typing import Optional
import uuid

app_name = "emotivoice"


def convert(src_wav, tgt_wav, tone_color_converter, get_se, save_path="./temp/output.wav", encode_message=""):
    try:
        # extract the tone color features of the source speaker and target speaker
        source_se, _ = get_se(src_wav, tone_color_converter, target_dir='processed', vad=True)
        target_se, _  = get_se(tgt_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        return {"error": f"Failed to extract speaker embedding: {e}"}
    tone_color_converter.convert(
        audio_src_path=src_wav, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)
    return save_path


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
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


router = AppInitializationRouter(app_name=app_name)

class TTSRequest(BaseModel):
    text_content: str = Field(..., description="要转换为语音的文本")
    speaker: str = Field('8051', description="说话人ID")
    emotion: Optional[str] = Field('', description="情感提示")

@router.post("/tts")
@change_dir(router.dir)
async def tts_api(request: TTSRequest):
    from repo.emotivoice.demo_page import tts
    _name = f'./temp/{str(uuid.uuid4())}.wav'
    tts(request.text_content, request.emotion, request.speaker, _name, router.models['models'], router.models['g2p'], router.models['lexicon'])
    with open(_name, 'rb') as file:
        audio_data = file.read()
    os.remove(_name)
    audio_base64 = base64.b64encode(audio_data).decode()
    return {"audio_base64": audio_base64}

class ConversionOnlyRequest(BaseModel):
    src_path: str = Field(..., description="要转换的语音内容")
    tgt_path: str = Field(..., description="要参考的目标音色")

@router.post("/convert_only")
@change_dir(router.dir)
async def conversion_only(request: ConversionOnlyRequest):
    from repo.emotivoice.tone_color_conversion import get_se
    save_path = convert(src_wav=request.src_path, tgt_wav=request.tgt_path, 
                        tone_color_converter=router.models['tone_color_converter'], get_se=get_se, encode_message='Airbox')
    if isinstance(save_path, dict):
        return {"text": save_path['error'], 'info': 'error message'}
    with open(save_path, 'rb') as file:
        audio_data = file.read()
    audio_base64 = base64.b64encode(audio_data).decode()
    return {"audio_base64": audio_base64}

class TTSWithConvertRequest(BaseModel):
    text_content: str = Field(..., description="要转换为语音的文本")
    speaker: str = Field('8051', description="原始说话人ID")
    emotion: Optional[str] = Field('', description="情感提示")
    tgt_path: str = Field(..., description="要参考的目标音色")

@router.post("/converttts")
@change_dir(router.dir)
async def tts_api(request: TTSRequest):    
    from repo.emotivoice.demo_page import tts
    from repo.emotivoice.tone_color_conversion import get_se

    src_wav = tts(request.text_content, request.emotion, request.speaker, f'./temp/{str(uuid.uuid4())}.wav',
                router.models['models'], router.models['g2p'], router.models['lexicon'])
    res_wav = convert(src_wav=src_wav, tgt_wav=request.tgt_path, 
                tone_color_converter=router.models['tone_color_converter'], get_se=get_se, encode_message='Airbox')
    if isinstance(res_wav, dict):
        return {"text": res_wav['error'], 'info': 'error message'}
    with open(res_wav, 'rb') as file:
        audio_data = file.read()
    audio_base64 = base64.b64encode(audio_data).decode()
    return {"audio_base64": audio_base64}