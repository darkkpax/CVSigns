$ErrorActionPreference = 'Stop'
Set-Location 'K:\Python\git\CVSigns'

& 'K:\Python\git\CVSigns\.venv-train\Scripts\yolo.exe' detect train `
  model=K:\Python\git\CVSigns\yolo12s.pt `
  data=K:\Python\git\CVSigns\train\yolo_rtsd\dataset.yaml `
  epochs=80 `
  imgsz=800 `
  batch=4 `
  device=0 `
  workers=6 `
  cache=disk `
  amp=True `
  patience=15 `
  cos_lr=True `
  close_mosaic=10 `
  project=K:\Python\git\CVSigns\runs\detect `
  name=rtsd_yolo12s_800_fast `
  2>&1 | Tee-Object K:\Python\git\CVSigns\train\logs\rtsd_yolo12s_800_fast.log
