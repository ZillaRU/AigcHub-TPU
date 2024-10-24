from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from api.base_api import BaseAPIRouter, change_dir, init_helper
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
class RoopFaceRequest(BaseModel):
    ## 有意义的兼容参数
    image: str = Field(..., description="Base64 encoded source image")

    ## 无意义的兼容参数
    model: Optional[str] = Field("roop_face", description="(形式参数无意义)")
    n : Optional[int] = Field(1, description="(形式参数无意义)")
    response_format: Optional[str] = Field("b64_json", description="(形式参数无意义)")
    size: Optional[str] = Field("1024x1024", description="(形式参数无意义)")
    user: Optional[str] = Field("", description="(形式参数无意义)")

    ## 专有参数
    target_img: Optional[str] = Field(..., description="Base64 encoded target image")
    restorer_visibility: float = Field(1.0, description="Visibility for the restorer")


@router.post("/v1/images/variations")
@change_dir(router.dir)
async def face_swap(request: RoopFaceRequest):
    src_image_bytes = BytesIO(base64.b64decode(request.image))
    src_image = Image.open(src_image_bytes)

    if request.target_img:
        tar_image_bytes = BytesIO(base64.b64decode(request.target_img))
        tar_image = Image.open(tar_image_bytes)
        result_image = swap_face(router.models['face_swapper'], src_image, tar_image)  # Assuming swap_face is a defined function
    else:
        numpy_image = np.array(src_image)
        numpy_image = router.models['restorer'].restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(src_image, restored_image, request.restorer_visibility)
        
    buffer = BytesIO()
    result_image.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    content = {
                "created": 1589478378, 
                "data": [
                    {
                        "b64_json": ret_img_b64
                    }
                        ]
                }
    return content