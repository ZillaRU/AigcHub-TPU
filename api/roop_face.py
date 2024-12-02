import base64
from io import BytesIO
from PIL import Image
import numpy as np
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from repo.roop_face.roop import swap_face, setup_model
from repo.roop_face.roop.inswappertpu import INSwapper
from typing import Optional

app_name = "roop_face"

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        self.models['face_swapper'] = INSwapper("./bmodel_files")
        self.models['restorer'] = setup_model('./bmodel_files/codeformer_1-3-512-512_1-235ms.bmodel')
        return {"message": f"Application {self.app_name} has been initialized successfully."}
    
    async def destroy_app(self):
        del self.models['face_swapper']
        del self.models['restorer']
    

router = AppInitializationRouter(app_name=app_name)

### 图像变换；兼容openai api，images/variations
@router.post("/v1/images/variations")
@change_dir(router.dir)
async def face_swap(
    image: UploadFile = File(...),
    target_img: UploadFile = File(...), #比openai多了一个参数
    restorer_visibility: Optional[float] = Form(1.0),
):
    src_image_bytes = await image.read()
    src_image = Image.open(BytesIO(src_image_bytes))
    tar_image_bytes = await target_img.read()
    tar_image = Image.open(BytesIO(tar_image_bytes))
    result_image = swap_face(router.models['face_swapper'], src_image, tar_image)
        
    buffer = BytesIO()
    result_image.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {"data": [{"b64_json": ret_img_b64}]}
    return JSONResponse(content=jsonable_encoder(content), media_type="application/json")


### 图像增强；兼容openai api，images/edit
@router.post("/v1/images/edit")
@change_dir(router.dir)
async def face_enhance(
    image: UploadFile = File(...),
    restorer_visibility: Optional[float] = Form(1.0),
):
    ori_image_bytes = await image.read()
    ori_image = Image.open(BytesIO(ori_image_bytes))
    print(f"Restore face with Codeformer")
    numpy_image = np.array(ori_image)
    numpy_image = router.models['restorer'].restore(numpy_image)
    restored_image = Image.fromarray(numpy_image)
    result_image = Image.blend(ori_image, restored_image, restorer_visibility)
    buffer = BytesIO()
    result_image.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {"data": [{"b64_json": ret_img_b64}]}
    return JSONResponse(content=jsonable_encoder(content), media_type="application/json")