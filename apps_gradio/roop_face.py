import gradio as gr

DESCRIPTION = "人像换脸/人像增强"
STANDALONE = True

gr.Markdown(f"## {DESCRIPTION}",  elem_id="welcome_message")


if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    from PIL import Image

    from repo.roop_face.roop import swap_face, setup_model
    from repo.roop_face.roop.inswappertpu import INSwapper


    os.chdir("repo/roop_face")

    face_swapper = INSwapper("./bmodel_files")
    restorer = setup_model('./bmodel_files/codeformer_1-3-512-512_1-235ms.bmodel')

    def func(source_img:Image.Image, target_img:Image.Image, use_enhance=True, restorer_visibility=1.0):
        src_img = source_img.convert('RGB')
        tar_img = target_img.convert('RGB')
        pil_res = swap_face(face_swapper, src_img, tar_img)
        if use_enhance:
            print(f"Restore face with Codeformer")
            numpy_image = np.array(pil_res)
            numpy_image = restorer.restore(numpy_image)
            restored_image = Image.fromarray(numpy_image)
            result_image = Image.blend(
                pil_res, restored_image, restorer_visibility
            )
            return result_image
        else:
            return pil_res

    title = f"<center><strong><font size='8'>人像换脸(●'◡'●) powered by sg2300x <font></strong></center>"
    css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

    with gr.Blocks(css=css, title="换脸") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(title)

        description_p = """ # 使用方法

                1. 上传人脸图像和目标图像，选择是否使用人像增强。
                2. 点击“换脸”。
                """
        with gr.Column():
            with gr.Row():
                img_input1 = gr.Image(label="人脸图像", sources=['upload'], type='pil')
                img_input2 = gr.Image(label="目标图像", sources=['upload'], type='pil')
                img_res = gr.Image(label="换脸图像", interactive=False)
            with gr.Row():
                use_enhance_orN = gr.Checkbox(label="人像增强", value=True)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        btn_p = gr.Button("换脸", variant="primary")
                        clear_btn_p = gr.Button("清空", variant="secondary")

            with gr.Column():
                gr.Markdown(description_p)

        btn_p.click(
            func, inputs=[img_input1, img_input2, use_enhance_orN], outputs=[img_res]
        )
        def clear():
            return [None, None, None]

        clear_btn_p.click(clear, outputs=[img_input1, img_input2, img_res])

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.queue().launch(server_port=args.port, inbrowser=True, server_name="0.0.0.0")
