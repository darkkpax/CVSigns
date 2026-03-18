---
title: Russian Traffic Signs YOLO12
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.47.0
app_file: app.py
pinned: false
license: mit
---

# Russian Traffic Signs YOLO12

Gradio Space for Russian traffic sign detection with a YOLO12 model trained on RTSD.

## Metrics

- Precision: `0.76584`
- Recall: `0.74393`
- mAP@0.50: `0.82171`
- mAP@0.50:0.95: `0.61105`

## Files

- `app.py`
- `requirements.txt`
- `best.pt`

## Local run

```bash
pip install -r requirements.txt
python app.py
```
