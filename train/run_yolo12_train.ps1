$ErrorActionPreference = 'Stop'
Set-Location 'K:\Python\git\CVSigns'
& 'K:\Python\git\CVSigns\.venv-train\Scripts\yolo.exe' detect train 
  model=K:\Python\git\CVSigns\yolo12n.pt 
  data=K:\Python\git\CVSigns\train\yolo_rtsd\dataset.yaml 
  epochs=100 
  imgsz=640 
  batch=8 
  device=0 
  workers=6 
  cache=disk 
  amp=True 
  patience=20 
  cos_lr=True 
  project=K:\Python\git\CVSigns\runs\detect 
  name=rtsd_yolo12n_640 
  2>&1 | Tee-Object K:\Python\git\CVSigns\train\logs\rtsd_yolo12n_640.log
