# 🚦 Traffic Sign UI (Next.js)

Web interface for the detection+classification pipeline trained by `../train.ipynb`.

## Run

1. start the model service (loads `runs/yolov8n_traffic_signs*/weights/best.pt` + `../models/gtsrb_cnn_scratch.pth`):

   ```
   # from the project root
   python server/infer_api.py          # → http://127.0.0.1:8001
   ```

2. install & start the UI:

   ```
   cd ui
   npm install
   npm run dev                         # → http://localhost:3000
   ```

3. open http://localhost:3000, drop an image, click **Detect signs**.

## 🎥 Live video (dashcam) mode

Switch to **🎥 Dashcam video**, drop an mp4/avi/webm, press **▶ Start live detection**:

- the browser samples ~9 frames/s from the playing video (hidden canvas → JPEG),
- frames go through the same `POST /infer` endpoint, so no extra server endpoint is needed,
- returned boxes (YOLO + CNN labels) are drawn live on an overlay canvas,
- the panel below shows a frame/sign counter, per-class totals and the current frame detections,
- the confidence slider (shared with photo mode) is passed as `?conf=` — forwarded by `app/api/infer/route.js`,
- **Sign scale (imgsz, 416 → 1280)** changes the detection resolution on the fly: 416 matches training for
  close-up footage; 960/1280 help when signs are small in large (1080p/4K) dashcam frames — for the bundled
  `database/video.mp4` (a natively 416×416 clip) keep 416 and lower confidence instead,
- detections are held for up to 3 consecutive "empty" frames, so boxes don't strobe on/off.

The upload goes through the Next.js route handler (`app/api/infer/route.js`) to the
Python service, so no CORS setup is needed. Set `MODEL_API_URL` env var to point at
a remote service instead of `http://127.0.0.1:8001`.
