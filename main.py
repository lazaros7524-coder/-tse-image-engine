<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<title>تست موتور تبدیل تصویر TSE</title>
<style>
  body { font-family: Tahoma, Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 22px; }
  .box { border: 2px dashed #999; padding: 30px; text-align: center; border-radius: 10px; margin: 20px 0; }
  button { background: #4f46e5; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 8px; cursor: pointer; }
  button:disabled { background: #aaa; }
  #status { margin-top: 15px; font-size: 15px; }
  #result { margin-top: 20px; text-align: center; }
  #result svg, #result img { max-width: 100%; border: 1px solid #ddd; background: white; }
  .row { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; margin-top: 10px; }
  .col { flex: 1; min-width: 250px; text-align: center; }
  input[type=range] { width: 200px; }
</style>
</head>
<body>
  <h1>🎨 تست موتور تبدیل تصویر به وکتور (TSE)</h1>
  <p>یک عکس انتخاب کنید و دکمه‌ی تبدیل را بزنید تا نتیجه را ببینید.</p>

  <div class="box">
    <input type="file" id="fileInput" accept="image/*"><br><br>
    <label>حساسیت لبه (پایین): <span id="lowVal">50</span></label><br>
    <input type="range" id="lowRange" min="10" max="200" value="50"><br>
    <label>حساسیت لبه (بالا): <span id="highVal">150</span></label><br>
    <input type="range" id="highRange" min="50" max="300" value="150"><br><br>
    <button id="submitBtn">تبدیل به استنسیل</button>
    <button id="depthBtn">تخمین نقشه‌ی عمق</button>
    <button id="shadeBtn">تولید هاشور سایه‌روشن</button>
  </div>

  <div id="status"></div>
  <div class="row">
    <div class="col">
      <h3>تصویر اصلی</h3>
      <div id="originalPreview"></div>
    </div>
    <div class="col">
      <h3>نتیجه (SVG)</h3>
      <div id="result"></div>
    </div>
  </div>

<script>
const SERVICE_URL = "https://tse-image-engine.onrender.com/vectorize";

const fileInput = document.getElementById('fileInput');
const submitBtn = document.getElementById('submitBtn');
const depthBtn = document.getElementById('depthBtn');
const shadeBtn = document.getElementById('shadeBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const originalPreview = document.getElementById('originalPreview');
const lowRange = document.getElementById('lowRange');
const highRange = document.getElementById('highRange');
const lowVal = document.getElementById('lowVal');
const highVal = document.getElementById('highVal');

lowRange.addEventListener('input', () => lowVal.textContent = lowRange.value);
highRange.addEventListener('input', () => highVal.textContent = highRange.value);

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  originalPreview.innerHTML = `<img src="${url}" style="max-width:100%;">`;
});

submitBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    statusEl.textContent = 'لطفاً اول یک عکس انتخاب کنید.';
    return;
  }

  submitBtn.disabled = true;
  statusEl.textContent = 'در حال ارسال و پردازش... (ممکن است سرور رایگان ۵۰ ثانیه‌ی اول کمی کند باشد)';
  resultEl.innerHTML = '';

  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams({
    low_threshold: lowRange.value,
    high_threshold: highRange.value
  });

  try {
    const response = await fetch(`${SERVICE_URL}?${params.toString()}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errText = await response.text();
      statusEl.textContent = `خطا: ${response.status} - ${errText}`;
      submitBtn.disabled = false;
      return;
    }

    const svgText = await response.text();
    resultEl.innerHTML = svgText;
    statusEl.textContent = '✅ با موفقیت انجام شد.';
  } catch (err) {
    statusEl.textContent = 'خطا در اتصال به سرویس: ' + err.message;
  }

  submitBtn.disabled = false;
});

depthBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    statusEl.textContent = 'لطفاً اول یک عکس انتخاب کنید.';
    return;
  }

  depthBtn.disabled = true;
  statusEl.textContent = 'در حال تخمین عمق... (اولین اجرا ممکن است به دلیل دانلود مدل ۱-۲ دقیقه طول بکشد)';
  resultEl.innerHTML = '';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch("https://tse-image-engine.onrender.com/depth", {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errText = await response.text();
      statusEl.textContent = `خطا: ${response.status} - ${errText}`;
      depthBtn.disabled = false;
      return;
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    resultEl.innerHTML = `<img src="${url}" style="max-width:100%;">`;
    statusEl.textContent = '✅ نقشه‌ی عمق با موفقیت تولید شد.';
  } catch (err) {
    statusEl.textContent = 'خطا در اتصال به سرویس: ' + err.message;
  }

  depthBtn.disabled = false;
});

shadeBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    statusEl.textContent = 'لطفاً اول یک عکس انتخاب کنید.';
    return;
  }

  shadeBtn.disabled = true;
  statusEl.textContent = 'در حال تولید هاشور سایه‌روشن...';
  resultEl.innerHTML = '';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch("https://tse-image-engine.onrender.com/shade", {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errText = await response.text();
      statusEl.textContent = `خطا: ${response.status} - ${errText}`;
      shadeBtn.disabled = false;
      return;
    }

    const svgText = await response.text();
    resultEl.innerHTML = svgText;
    statusEl.textContent = '✅ هاشور با موفقیت تولید شد.';
  } catch (err) {
    statusEl.textContent = 'خطا در اتصال به سرویس: ' + err.message;
  }

  shadeBtn.disabled = false;
});
</script>
</body>
</html>
