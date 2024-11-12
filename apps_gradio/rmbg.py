import gradio as gr

try:
    from skimage import io
    import torch
    from PIL import Image
    from repo.rmbg.python.utilities import preprocess_image, postprocess_image
    from repo.rmbg.python.npuengine import EngineOV

    STANDALONE = False

    net = EngineOV("repo/rmbg/models/rmbg.bmodel", device_id=0)

    def process_image(im_path):
        # Prepare input
        model_input_size = [1024, 1024]
        orig_im = io.imread(im_path)
        orig_im_size = orig_im.shape[0:2]
        image = preprocess_image(orig_im, model_input_size)

        # Inference 
        result = net([image.numpy()])[0]
        result = torch.from_numpy(result).float() 
        
        # Post process    
        result_image = postprocess_image(result, orig_im_size)

        # Convert result to image
        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
        orig_image = Image.open(im_path)
        no_bg_image.paste(orig_image, mask=pil_im)
        
        return orig_image, no_bg_image

    def gradio_blocks():
        with gr.Blocks() as demo:
            gr.Markdown("# RMBG on AirBox")
            with gr.Row():
                with gr.Column():
                    inp_image = gr.Image(type="filepath", label="Upload your image")
                    submit_button = gr.Button("Remove Background")
                with gr.Column():
                    original_image = gr.Image(label="Original Image")
                    output_image = gr.Image(label="Processed Image")
            
            submit_button.click(process_image, inputs=inp_image, outputs=[original_image, output_image])

        return demo

    demo = gradio_blocks()
except:
    gr.Markdown(f"## Not initialized",  elem_id="welcome_message")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.queue().launch(server_port=args.port,inbrowser=True)

