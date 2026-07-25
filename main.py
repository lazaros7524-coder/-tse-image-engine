import subprocess
import tempfile
import os

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse

app = FastAPI(title="TSE Image Engine")


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
