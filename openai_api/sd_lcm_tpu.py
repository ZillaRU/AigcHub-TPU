import io
import base64
import random
from PIL import Image
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper
import os, re
from typing import Optional
from fastapi import File, Form, UploadFile


TEST=False
DEVICE_ID=os.environ.get('DEVICE_ID', 0)
BASENAME = os.environ.get('BASENAME', 'awportraitv14')
CONTROLNET = os.environ.get('CONTROLNET', '')
RETURN_BASE64 = bool(int(os.environ.get('RETURN_BASE64', 1)))

def handle_base64_image(controlnet_image):
    # 目前只支持一个controlnet_image, 不可以是list
    if isinstance(controlnet_image, list):
        controlnet_image = controlnet_image[0]
    if controlnet_image.startswith("data:image"):
        controlnet_image = controlnet_image.split(",")[1]
        
    return controlnet_image

def handle_output_base64_image(image_base64):
    if not RETURN_BASE64:
        return image_base64
    if not image_base64.startswith("data:image"):
        image_base64 = "data:image/jpeg;base64," + image_base64
    return image_base64

def get_shape_by_ratio(width, height):
    ratio_shape = {
        1:[512,512],
        2/3:[640,960],
        3/2:[960,640],
        4/3:[704,896],
        3/4:[896,704],
        9/16:[576,1024],
        16/9:[1024,576],
    }
    ratio = width/height
    # 这个ratio找到最接近的ratio_shape
    ratio_shape_list = list(ratio_shape.keys())
    ratio_shape_list.sort(key=lambda x:abs(x-ratio))
    nshape = ratio_shape[ratio_shape_list[0]]
    print(nshape)
    return nshape

app_name = "sd_lcm_tpu"

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        # ori_dir = os.getcwd()
        # sys.path.append(os.path.join(ori_dir, self.dir))
        # os.chdir(self.dir)
        from repo.sd_lcm_tpu.sd import StableDiffusionPipeline

        self.models['pipeline'] = StableDiffusionPipeline(basic_model="hellonijicute",
                                                          controlnet_name="",
                                                          scheduler="LCM")
        # os.chdir(ori_dir)
        return {"message": f"Application {self.app_name} has been initialized successfully."}
    
    async def destroy_app(self):
        del self.models['pipeline']
        return {"message": f"Application {self.app_name} has been destroyed successfully."}


router = AppInitializationRouter(app_name=app_name)


### 00 文本转图像，兼容openai api，image/genrations
@router.post("/v1/images/generations")
@change_dir(router.dir)
async def txt2img(
    prompt: str = Form(...),
    size: Optional[str] = Form("512x512"),
    negative_prompt: Optional[str] = Form(None),
    num_inference_steps: Optional[int] = Form(5),
    guidance_scale: Optional[float] = Form(1.0),
    strength: Optional[float] = Form(0.8),
    seed: Optional[int] = Form(-1),
    sampler_index: Optional[str] = Form("LCM")
):
    if seed == -1:
        seed = random.randint(0, 2 ** 31 - 1)
    subseed = seed # 不可以为-1
    subseed_strength = 0.0
    seed_resize_from_h = 1
    seed_resize_from_w = 1

    match = re.match(r'^(\d+)\s*(x|\*|×|by|times|,)\s*(\d+)$', size.strip(), re.IGNORECASE)
    height, width = map(int, match.groups()[0::2])
    
    nwidth, nheight = get_shape_by_ratio(width, height)
    router.models['pipeline'].set_height_width(nwidth, nheight)
    controlnet_image = None
    init_image = None
    mask = None
    controlnet_args = {}

    try:
        if sampler_index != "LCM":
            num_inference_steps = max(num_inference_steps, 20)
        router.models['pipeline'].scheduler = sampler_index
        img_pil = router.models['pipeline'](
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
                scheduler=sampler_index,
            )
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')
        ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content = {
            "data": [
                {
                    "b64_json": ret_img_b64
                }
                    ]
            }
        return JSONResponse(content=jsonable_encoder(content), media_type="application/json")
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({'ret_img': [], 'message': str(e)}), media_type="application/json")


### 01 图生图，兼容openai api，images/edits
@router.post("/v1/images/edits")
@change_dir(router.dir)
async def img2img(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    size: str = Form("512x512"),
    negative_prompt: Optional[str] = Form(None),
    num_inference_steps: Optional[int] = Form(5),
    guidance_scale: Optional[float] = Form(1.0),
    strength: Optional[float] = Form(0.8),
    seed: Optional[int] = Form(-1),
    sampler_index: Optional[str] = Form("LCM")  
):
    """img2img"""
    # 从 JSON 数据中获取所需数据
    if seed == -1:
        seed = random.randint(0, 2 ** 31 - 1)
    subseed = seed # 不可以为-1
    subseed_strength = 0.0
    seed_resize_from_h = 1
    seed_resize_from_w = 1

    match = re.match(r'^(\d+)\s*(x|\*|×|by|times|,)\s*(\d+)$', size.strip(), re.IGNORECASE)
    height, width = map(int, match.groups()[0::2])
    
    nwidth, nheight = get_shape_by_ratio(width, height)
    router.models['pipeline'].set_height_width(nwidth, nheight)
    controlnet_image = None
    init_image = None
    mask = None
    controlnet_args = {}

    ori_image_bytes = await image.read()
    init_image = Image.open(io.BytesIO(ori_image_bytes))

    try:
        if sampler_index != "LCM":
            num_inference_steps = max(num_inference_steps, 20)
        router.models['pipeline'].scheduler = sampler_index
        img_pil = router.models['pipeline'](
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
                scheduler=sampler_index,
            )
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')
        ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content = {
            "data": [
                {
                    "b64_json": ret_img_b64
                }
                    ]
            }
        return JSONResponse(content=jsonable_encoder(content), media_type="application/json")
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({'ret_img': [], 'message': str(e)}), media_type="application/json")


#### 02 图像超分，兼容openai api，images/variations
@router.post("/v1/images/variations")
@change_dir(router.dir)
async def upscale(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    num_inference_steps: Optional[int] = Form(5),
    guidance_scale: Optional[float] = Form(1.0),
    strength: Optional[float] = Form(0.8),
    seed: Optional[int] = Form(-1),
    upscale_by: Optional[int] = Form(2),
    sampler_index: Optional[str] = Form("LCM")
):
    if seed == -1:
        seed = random.randint(0, 2 ** 31 - 1)
    mask = None
    subseed = seed
    subseed_strength = 0.0
    seed_resize_from_h = 1
    seed_resize_from_w = 1
    upscale_factor = upscale_by
    
    ori_image_bytes = await image.read()
    init_image = Image.open(io.BytesIO(ori_image_bytes))

    controlnet_image = None
    controlnet_args  = {}
    
    upscale_type   = 'LINEAR' # 必须大写 只有两种形式 LINEAR 和 CHESS
    tile_width     = 512 # 目前tile大小规定为512 多tile的方式需要再测试
    tile_height    = 512 # 目前tile大小规定为512 多tile的方式需要再测试
    mask_blur      = 8.0
    padding        = 12
    upscaler       = None # placeholder 用于以后的超分模型
    seams_fix      = {}
    seams_fix_enable= False
    
    try:
        if sampler_index != "LCM":
            num_inference_steps = max(num_inference_steps, 20)
        router.models['pipeline'].scheduler = sampler_index
        image = router.models['pipeline'].wrap_upscale(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            mask=mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_img = controlnet_image,
            seeds = [seed],
            subseeds = [subseed],
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            controlnet_args = controlnet_args,
            # upscale 参数
            upscale_factor = upscale_factor,
            upscale_type = upscale_type,
            mask_blur = mask_blur,
            tile_width = tile_width,
            tile_height = tile_height,
            padding   = padding,
            seams_fix_enable = seams_fix_enable,
            upscaler = upscaler,
            seams_fix = seams_fix
        )
        img_pil = image
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')
        ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        ret_img_b64 = handle_output_base64_image(ret_img_b64)
        content = {
            "data": [
                {
                    "b64_json": ret_img_b64
                }
                    ]
            }
        return JSONResponse(content)
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(trace)
        print(e)
        print("error")
        return JSONResponse({'code': 1, 'message': str(e)})
