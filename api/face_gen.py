from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from api.base_api import BaseAPIRouter
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from repo.roop_face.roop import swap_face, setup_model
from repo.roop_face.roop.inswappertpu import INSwapper
import os

app_name = "roop_face"
router = BaseAPIRouter(app_name=app_name)

@router.post("/initialize")
async def initialize_app():
    ori_dir = os.getcwd()
    os.chdir(f"repo/{app_name}")
    router.models['face_swapper'] = INSwapper("./bmodel_files")
    router.models['restorer'] = setup_model('./bmodel_files/codeformer_1-3-512-512_1-235ms.bmodel')
    os.chdir(ori_dir)
    # # 返回成功信息
    return {"message": f"Application {app_name} has been initialized successfully."}

class FaceSwapRequest(BaseModel):
    source_img: str = Field(..., description="Base64 encoded source image")
    target_img: str = Field(..., description="Base64 encoded target image")
    payload: dict = Field(..., description="Additional payload")

class FaceEnhanceRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image to enhance")
    restorer_visibility: float = Field(1.0, description="Visibility for the restorer")
    payload: dict = Field(..., description="Additional payload")

@router.post("/face_swap")
async def face_swap(request: FaceSwapRequest):
    ori_dir = os.getcwd()
    os.chdir(f"repo/{app_name}")
    src_image_bytes = BytesIO(base64.b64decode(request.source_img))
    src_image = Image.open(src_image_bytes)
    tar_image_bytes = BytesIO(base64.b64decode(request.target_img))
    tar_image = Image.open(tar_image_bytes)
    pil_res = swap_face(router.models['face_swapper'], src_image, tar_image)  # Assuming swap_face is a defined function
    buffer = BytesIO()
    pil_res.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {'ret_img': [ret_img_b64], 'message': 'success'}
    os.chdir(ori_dir)
    return JSONResponse(content=jsonable_encoder(content), media_type="application/json")

@router.post("/face_enhance")
async def face_enhance(request: FaceEnhanceRequest):
    ori_image_bytes = BytesIO(base64.b64decode(request.image))
    ori_image = Image.open(ori_image_bytes)
    print(f"Restore face with Codeformer")
    numpy_image = np.array(ori_image)
    numpy_image = router.models['restorer'].restore(numpy_image)
    restored_image = Image.fromarray(numpy_image)
    result_image = Image.blend(ori_image, restored_image, request.restorer_visibility)
    buffer = BytesIO()
    result_image.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    content = {'ret_img': [ret_img_b64], 'message': 'success'}
    return JSONResponse(content=jsonable_encoder(content), media_type="application/json")
