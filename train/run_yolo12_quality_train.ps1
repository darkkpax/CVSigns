$ErrorActionPreference = 'Stop'
Set-Location 'K:\Python\git\CVSigns'

& 'K:\Python\git\CVSigns\.venv-train\Scripts\yolo.exe' detect train `
  model=K:\Python\git\CVSigns\yolo12s.pt `
  data=K:\Python\git\CVSigns\train\yolo_rtsd\dataset.yaml `
  epochs=150 `
  imgsz=960 `
  batch=4 `
  device=0 `
  workers=6 `
  cache=disk `
  amp=True `
  patience=30 `
  cos_lr=True `
  close_mosaic=10 `
  project=K:\Python\git\CVSigns\runs\detect `
  name=rtsd_yolo12s_960_quality `
  2>&1 | Tee-Object K:\Python\git\CVSigns\train\logs\rtsd_yolo12s_960_quality.log
