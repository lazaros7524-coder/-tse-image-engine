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

def compute_depth_map(img_bgr):
    """نقشه‌ی عمق نرمال‌شده (بین ۰ تا ۱) را در ابعاد اصلی تصویر برمی‌گرداند."""
    orig_h, orig_w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        img_rgb, (DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE), interpolation=cv2.INTER_CUBIC
    )
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    session = get_depth_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    depth_map = np.squeeze(outputs[0])
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    if d_max - d_min > 1e-6:
        depth_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_map)

    return cv2.resize(depth_norm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


import re


def make_potrace_svg_fragment(bitmap_uint8, work_w, work_h, turdsize=8):
    """بیت‌مپ را به SVG وکتوری تبدیل می‌کند و ابعاد را با فضای مختصات پیکسلی هماهنگ می‌کند.
    turdsize: حذف لکه/خط‌های کوچک‌تر از این تعداد پیکسل (کاهش نویز و شلوغی)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bmp_path = os.path.join(tmp_dir, "edges.bmp")
        svg_path = os.path.join(tmp_dir, "edges.svg")
        cv2.imwrite(bmp_path, bitmap_uint8)

        result = subprocess.run(
            ["potrace", bmp_path, "-s", "-t", str(turdsize), "-o", svg_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        with open(svg_path, "r", encoding="utf-8") as f:
            svg_text = f.read()

    start = svg_text.find("<svg")
    end = svg_text.rfind("</svg>") + len("</svg>")
    svg_body = svg_text[start:end]

    # واحد pt خروجی potrace را با واحد بدون‌واحد (پیکسل) فضای ترکیبی هماهنگ می‌کنیم
    svg_body = re.sub(r'width="[^"]*"', f'width="{work_w}"', svg_body, count=1)
    svg_body = re.sub(r'height="[^"]*"', f'height="{work_h}"', svg_body, count=1)
    return svg_body


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
    low_threshold: int = Query(60, description="آستانه‌ی پایین Canny"),
    high_threshold: int = Query(160, description="آستانه‌ی بالای Canny"),
    smoothing: int = Query(3, description="تعداد دفعات اجرای فیلتر هموارساز (حذف بافت ریز مثل ریش/مو)"),
    turdsize: int = Query(20, description="حذف خط/لکه‌های کوچک‌تر از این مقدار (پیکسل) برای تمیزتر شدن طرح"),
    max_dimension: int = Query(
        400, description="حداکثر عرض/ارتفاع برای لبه‌یابی؛ کوچک‌تر یعنی خطوط ساده‌تر و کمتر شلوغ"
    ),
    invert: bool = Query(False, description="اگر خطوط باید سفید روی سیاه باشند"),
):
    # ۱. خواندن تصویر آپلودشده
    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "تصویر قابل خواندن نیست"})

    # ۲. کوچک‌کردن ابعاد قبل از لبه‌یابی — مهم‌ترین عامل برای حذف شلوغی بافت‌های ریز (مثل ریش/مو)
    h0, w0 = img.shape[:2]
    scale = min(1.0, max_dimension / max(w0, h0))
    work_w, work_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    img_small = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_AREA)

    # ۳. حذف بافت‌های ریز باقی‌مانده با فیلتر دوطرفه، بدون از بین بردن لبه‌های اصلی
    smoothed = img_small.copy()
    for _ in range(max(0, smoothing)):
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)

    blurred = cv2.GaussianBlur(smoothed, (3, 3), 0)

    # ۴. تشخیص لبه با Canny
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # ۵. آماده‌سازی بیت‌مپ برای Potrace (خطوط سیاه روی زمینه‌ی سفید)
    bitmap = cv2.bitwise_not(edges) if not invert else edges

    try:
        svg_content = make_potrace_svg_fragment(bitmap, work_w, work_h, turdsize=turdsize)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "خطا در اجرای potrace", "details": str(e)},
        )

    return Response(content=svg_content, media_type="image/svg+xml")


@app.post("/depth")
async def depth(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "تصویر قابل خواندن نیست"})

    try:
        depth_norm = compute_depth_map(img_bgr)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "خطا در اجرای مدل عمق", "details": str(e)},
        )

    depth_img = (depth_norm * 255).astype(np.uint8)
    success, png_bytes = cv2.imencode(".png", depth_img)
    if not success:
        return JSONResponse(status_code=500, content={"error": "خطا در تولید خروجی"})

    return Response(content=png_bytes.tobytes(), media_type="image/png")


@app.post("/shade")
async def shade(
    file: UploadFile = File(...),
    row_spacing: int = Query(5, description="فاصله‌ی بین ردیف‌های هاشور (پیکسل)"),
    dash_length: int = Query(3, description="طول هر تکه‌خط هاشور (پیکسل)"),
    max_dimension: int = Query(700, description="حداکثر عرض/ارتفاع کاری برای سرعت بیشتر"),
    curve_strength: float = Query(15.0, description="میزان انحنای خطوط متناسب با فرم عمق"),
    depth_weight: float = Query(
        0.35,
        description="سهم نقشه‌ی عمق در تراکم هاشور (۰ تا ۱)؛ باقی از روشنایی خود عکس (جزئیات ریز) گرفته می‌شود",
    ),
    invert: bool = Query(False, description="اگر نواحی روشن باید پررنگ شوند به‌جای نواحی تیره"),
):
    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "تصویر قابل خواندن نیست"})

    try:
        depth_norm = compute_depth_map(img_bgr)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "خطا در اجرای مدل عمق", "details": str(e)},
        )

    orig_h, orig_w = depth_norm.shape[:2]

    # ۱. کوچک‌کردن به یک ابعاد کاری برای سرعت و تعداد معقول خط‌ها
    scale = min(1.0, max_dimension / max(orig_w, orig_h))
    work_w, work_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
    depth_work = cv2.resize(depth_norm, (work_w, work_h), interpolation=cv2.INTER_AREA)

    # ۲. روشنایی خودِ عکس اصلی برای جزئیات ریز (چروک، سایه‌ی زیر چشم و ...)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_work = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray_work = cv2.GaussianBlur(gray_work, (0, 0), sigmaX=1.5)

    # ۳. نسخه‌ی هموارشده از عمق فقط برای تعیین انحنای کلی خطوط (فرم بزرگ)
    depth_smooth = cv2.GaussianBlur(depth_work, (0, 0), sigmaX=max(1, row_spacing))

    # ۴. ترکیب «میزان جوهر» از عمق (فرم کلی) و روشنایی عکس (جزئیات ریز)
    dw = float(np.clip(depth_weight, 0.0, 1.0))
    ink_depth = depth_work if invert else (1.0 - depth_work)
    ink_luma = gray_work if invert else (1.0 - gray_work)
    ink_map = dw * ink_depth + (1.0 - dw) * ink_luma

    # ۵. تولید هاشور کانتورمحور (منحنی بر اساس فرم عمق، تراکم بر اساس ترکیب عمق+روشنایی)
    svg_lines = []
    for y0 in range(0, work_h, row_spacing):
        y_curve = y0 + curve_strength * (depth_smooth[min(y0, work_h - 1), :] - 0.5)
        y_curve = np.clip(y_curve, 0, work_h - 1)

        error = 0.0
        x = 0
        while x < work_w:
            yi = min(int(round(y_curve[x])), work_h - 1)
            error += float(ink_map[yi, x])
            if error >= 1.0:
                error -= 1.0
                x_end = min(x + dash_length, work_w)
                y1 = y_curve[x]
                y2 = y_curve[min(x_end - 1, work_w - 1)]
                svg_lines.append(
                    f'<line x1="{x}" y1="{y1:.1f}" x2="{x_end}" y2="{y2:.1f}" '
                    f'stroke="black" stroke-width="1"/>'
                )
                x = x_end
            else:
                x += 1

    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {work_w} {work_h}" '
        f'width="{work_w}" height="{work_h}">'
        f'<rect width="{work_w}" height="{work_h}" fill="white"/>'
        + "".join(svg_lines)
        + "</svg>"
    )

    return Response(content=svg_content, media_type="image/svg+xml")


@app.post("/stencil")
async def stencil(
    file: UploadFile = File(...),
    low_threshold: int = Query(50, description="آستانه‌ی پایین Canny برای خط اصلی"),
    high_threshold: int = Query(150, description="آستانه‌ی بالای Canny برای خط اصلی"),
    row_spacing: int = Query(5, description="فاصله‌ی بین ردیف‌های هاشور (پیکسل)"),
    dash_length: int = Query(3, description="طول هر تکه‌خط هاشور (پیکسل)"),
    max_dimension: int = Query(700, description="حداکثر عرض/ارتفاع کاری برای سرعت بیشتر"),
    curve_strength: float = Query(15.0, description="میزان انحنای خطوط هاشور متناسب با فرم عمق"),
    depth_weight: float = Query(
        0.35, description="سهم نقشه‌ی عمق در تراکم هاشور (۰ تا ۱)؛ باقی از روشنایی خود عکس گرفته می‌شود"
    ),
    smoothing: int = Query(3, description="تعداد دفعات فیلتر هموارساز روی خط اصلی (حذف بافت ریز)"),
    turdsize: int = Query(20, description="حذف خط/لکه‌های کوچک‌تر از این مقدار (پیکسل) در خط اصلی"),
    lineart_dimension: int = Query(
        400, description="ابعاد کاری مخصوص خط اصلی؛ کوچک‌تر یعنی خط ساده‌تر و کمتر شلوغ"
    ),
):
    """خروجی نهایی و کامل: خط اصلی طرح (لاین‌آرت) + هاشور سایه‌روشن، در یک SVG واحد."""
    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "تصویر قابل خواندن نیست"})

    try:
        depth_norm = compute_depth_map(img_bgr)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "خطا در اجرای مدل عمق", "details": str(e)},
        )

    orig_h, orig_w = depth_norm.shape[:2]
    scale = min(1.0, max_dimension / max(orig_w, orig_h))
    work_w, work_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))

    depth_work = cv2.resize(depth_norm, (work_w, work_h), interpolation=cv2.INTER_AREA)
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_work = cv2.resize(gray_full, (work_w, work_h), interpolation=cv2.INTER_AREA)

    # ۱. خط اصلی طرح: ابتدا در ابعاد کوچک‌تر پردازش می‌شود تا خطوط ساده و کم‌شلوغ بمانند،
    #    سپس با تغییر ویژگی‌های width/height به بوم مشترک با هاشور هم‌تراز می‌شود (viewBox خودش مقیاس می‌گیرد)
    la_scale = min(1.0, lineart_dimension / max(work_w, work_h))
    la_w, la_h = max(1, int(work_w * la_scale)), max(1, int(work_h * la_scale))
    gray_lineart = cv2.resize(gray_work, (la_w, la_h), interpolation=cv2.INTER_AREA)

    smoothed = gray_lineart.copy()
    for _ in range(max(0, smoothing)):
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)
    blurred = cv2.GaussianBlur(smoothed, (3, 3), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    bitmap = cv2.bitwise_not(edges)
    try:
        lineart_svg_fragment = make_potrace_svg_fragment(bitmap, work_w, work_h, turdsize=turdsize)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "خطا در اجرای potrace", "details": str(e)},
        )

    # ۲. هاشور سایه‌روشن (همان روش مرحله‌ی قبل)
    gray_norm = gray_work.astype(np.float32) / 255.0
    gray_blur = cv2.GaussianBlur(gray_norm, (0, 0), sigmaX=1.5)
    depth_smooth = cv2.GaussianBlur(depth_work, (0, 0), sigmaX=max(1, row_spacing))

    dw = float(np.clip(depth_weight, 0.0, 1.0))
    ink_depth = 1.0 - depth_work
    ink_luma = 1.0 - gray_blur
    ink_map = dw * ink_depth + (1.0 - dw) * ink_luma

    svg_lines2 = []
    for y0 in range(0, work_h, row_spacing):
        y_curve = y0 + curve_strength * (depth_smooth[min(y0, work_h - 1), :] - 0.5)
        y_curve = np.clip(y_curve, 0, work_h - 1)

        error = 0.0
        x = 0
        while x < work_w:
            yi = min(int(round(y_curve[x])), work_h - 1)
            error += float(ink_map[yi, x])
            if error >= 1.0:
                error -= 1.0
                x_end = min(x + dash_length, work_w)
                y1 = y_curve[x]
                y2 = y_curve[min(x_end - 1, work_w - 1)]
                svg_lines2.append(
                    f'<line x1="{x}" y1="{y1:.1f}" x2="{x_end}" y2="{y2:.1f}" '
                    f'stroke="black" stroke-width="0.6" stroke-opacity="0.75"/>'
                )
                x = x_end
            else:
                x += 1

    # ۳. ترکیب نهایی: زمینه‌ی سفید + هاشور (زیر) + خط اصلی طرح (رو)
    combined_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {work_w} {work_h}" '
        f'width="{work_w}" height="{work_h}">'
        f'<rect width="{work_w}" height="{work_h}" fill="white"/>'
        + "".join(svg_lines2)
        + lineart_svg_fragment
        + "</svg>"
    )

    return Response(content=combined_svg, media_type="image/svg+xml")
