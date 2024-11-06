#!/bin/bash
source hub_venv/bin/activate
export NO_ALBUMENTATIONS_UPDATE=1
python main_gradio_hub.py