#!/bin/bash
source hub_venv/bin/activate
export NO_ALBUMENTATIONS_UPDATE=1
python main_hub.py --host 0.0.0.0 --port 8000 "$@"