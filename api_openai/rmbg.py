import base64
from io import BytesIO
from PIL import Image
import numpy as np
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from repo.rmbg.python.utilities import preprocess_image, postprocess_image
import torch
from repo.rmbg.python.npuengine import EngineOV

app_name = "rmbg"

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        self.model = EngineOV("models/rmbg.bmodel", device_id=0)
        return {"message": f"Application {self.app_name} has been initialized successfully."}
    
    async def destroy_app(self):
        del self.model
    

router = AppInitializationRouter(app_name=app_name)


### 图像去背景；兼容openai api，images/edit
@router.post("/v1/images/edit")
@change_dir(router.dir)
async def remove_background(
    image: UploadFile = File(...),
):

    ori_image_bytes = await image.read()
    ori_image_data = BytesIO(ori_image_bytes)
    ori_image = Image.open(ori_image_data)
    image_np = np.array(ori_image)
    orig_im_size = image_np.shape[:2]
    model_input_size = [1024, 1024]
    image = preprocess_image(image_np, model_input_size)
    
    # Inference
    result = router.model([image.numpy()])[0]
    result = torch.from_numpy(result).float()
    
    # Post-process
    result_image = postprocess_image(result, orig_im_size)

    # Creating no background image
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(ori_image_data)
    no_bg_image.paste(orig_image, mask=pil_im)
    
    # Convert to b64_json for output
    buffer = BytesIO()
    no_bg_image.save(buffer, format='PNG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {"data": [{"b64_json": ret_img_b64}]}
    return JSONResponse(content=jsonable_encoder(content), media_type="application/json")
    