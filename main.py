import subprocess
import tempfile
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

# مدل سبک و از پیش کوانتیزه‌ی Depth Anything V2 (Small) — حدود ۵۰ مگابایت
DEPTH_MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/"
    "resolve/main/onnx/model_quantized.onnx"
)
DEPTH_MODEL_PATH = "/tmp/depth_model.onnx"
DEPTH_INPUT_SIZE = 518  # اندازه‌ی ورودی استاندارد این مدل

_depth_session = None


def get_depth_session():
    """مدل را فقط یک‌بار (در اولین درخواست) دانلود و بارگذاری می‌کند."""
    global _depth_session
    if _depth_session is None:
        if not os.path.exists(DEPTH_MODEL_PATH):
            urllib.request.urlretrieve(DEPTH_MODEL_URL, DEPTH_MODEL_PATH)
        _depth_session = ort.InferenceSession(
            DEPTH_MODEL_PATH, providers=["CPUExecutionProvider"]
        )
    return _depth_session

app = FastAPI(title="TSE Image Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/vectorize")
async def vectorize(
    file: UploadFile = File(...),
    low_threshold: int = Query(50, description="آستانه‌ی پایین Canny"),
    high_threshold: int = Query(150, description="آستانه‌ی بالای Canny"),
    invert: bool = Query(False, description="اگر خطوط باید سفید روی سیاه باشند"),
):
    # ۱. خواندن تصویر آپلودشده
    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "تصویر قابل خواندن نیست"})

    # ۲. کاهش نویز قبل از تشخیص لبه
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # ۳. تشخیص لبه با Canny
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # ۴. آماده‌سازی بیت‌مپ برای Potrace (خطوط سیاه روی زمینه‌ی سفید)
    bitmap = cv2.bitwise_not(edges) if not invert else edges

    with tempfile.TemporaryDirectory() as tmp_dir:
        bmp_path = os.path.join(tmp_dir, "edges.bmp")
        svg_path = os.path.join(tmp_dir, "edges.svg")
        cv2.imwrite(bmp_path, bitmap)

        # ۵. فراخوانی Potrace برای تبدیل بیت‌مپ به SVG وکتوری
        result = subprocess.run(
            ["potrace", bmp_path, "-s", "-o", svg_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": "خطا در اجرای potrace", "details": result.stderr},
            )

        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

    return Response(content=svg_content, media_type="image/svg+xml")


@app.post("/depth")
async def depth(file: UploadFile = File(...)):
    # ۱. خواندن تصویر آپلودشده
    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "تصویر قابل خواندن نیست"})

    orig_h, orig_w = img_bgr.shape[:2]

    # ۲. آماده‌سازی ورودی مدل (تغییر اندازه + نرمال‌سازی استاندارد ImageNet)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        img_rgb, (DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE), interpolation=cv2.INTER_CUBIC
    )
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    # ۳. اجرای مدل
    try:
        session = get_depth_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "خطا در اجرای مدل عمق", "details": str(e)},
        )

    depth_map = np.squeeze(outputs[0])

    # ۴. نرمال‌سازی خروجی به یک تصویر grayscale قابل مشاهده
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    if d_max - d_min > 1e-6:
        depth_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_map)
    depth_img = (depth_norm * 255).astype(np.uint8)

    # ۵. بازگرداندن به ابعاد اصلی تصویر ورودی
    depth_img = cv2.resize(depth_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    success, png_bytes = cv2.imencode(".png", depth_img)
    if not success:
        return JSONResponse(status_code=500, content={"error": "خطا در تولید خروجی"})

    return Response(content=png_bytes.tobytes(), media_type="image/png")
