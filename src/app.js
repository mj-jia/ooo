import * as ort from '../vendor/ort.all.min.mjs';

/* ---------------- 模型配置 ---------------- */
const MODELS = {
  fast: {
    url: './models/realesr-general-x4v3.onnx',
    input: 'input',
    output: 'output',
    label: '均衡 · x4v3',
    bytes: '4.7 MB',
    maxInput: 1280,
  },
  high: {
    url: './models/upscaler-x4.onnx',
    input: 'rgb',
    output: 'rgb_out',
    label: '高质量 · x4plus',
    bytes: '66 MB',
    maxInput: 640,
  },
};

const SCALE = 4;   // 两个模型都是 4 倍
const TILE = 256;  // 单块输入边长
const MARGIN = 24; // 每块向四周多读的上下文

ort.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
ort.env.logLevel = 'error';

/* ---------------- DOM ---------------- */
const $ = (id) => document.getElementById(id);
const dropEl = $('drop');
const fileEl = $('file');
const workEl = $('work');
const cvBefore = $('cvBefore');
const cvAfter = $('cvAfter');
const handleEl = $('handle');
const sliderEl = $('slider');
const runBtn = $('run');
const dlBtn = $('download');
const againBtn = $('again');
const scaleSel = $('scale');
const modelSel = $('model');
const modelNote = $('modelNote');
const progEl = $('progress');
const barEl = $('bar');
const ptextEl = $('ptext');
const errEl = $('err');
const mName = $('mName'), mSize = $('mSize'), mOut = $('mOut'), mDev = $('mDev'), mTime = $('mTime');

/* ---------------- 状态 ---------------- */
const sessions = {};        // key -> { sess, input, output, ep }
let currentModel = 'fast';
let srcBitmap = null;
let resultBlob = null;
let busy = false;

/* ---------------- 工具 ---------------- */
const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

// 镜像反射索引：越界时折回图内
function reflect(i, n) {
  if (n <= 1) return 0;
  const period = 2 * n - 2;
  let j = i % period;
  if (j < 0) j += period;
  return j < n ? j : period - j;
}

function showErr(msg) {
  errEl.textContent = msg;
  errEl.hidden = false;
}

function clearErr() {
  errEl.hidden = true;
  errEl.textContent = '';
}

function setProgress(ratio, text) {
  progEl.hidden = false;
  barEl.style.width = `${Math.round(ratio * 100)}%`;
  ptextEl.textContent = text;
}

function setBusy(v) {
  busy = v;
  runBtn.disabled = v;
  againBtn.disabled = v;
  dlBtn.disabled = v || !resultBlob;
}

function updateModelNote() {
  const m = MODELS[currentModel];
  modelNote.textContent = `${m.bytes} · 输入长边上限 ${m.maxInput} 像素`;
}

/* ---------------- 模型加载 ---------------- */
async function fetchModelBuffer(url, onProgress) {
  try {
    const cache = await caches.open('upsr-model-v2');
    const hit = await cache.match(url);
    if (hit) {
      onProgress?.(1, '已从本地缓存读取模型');
      return await hit.arrayBuffer();
    }
  } catch (_) { /* 浏览器不支持 Cache API 时忽略 */ }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`模型下载失败：HTTP ${res.status}`);

  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const mb = (received / 1048576).toFixed(1);
    const pct = total ? received / total : 0;
    onProgress?.(pct, `下载模型 ${mb} MB${total ? ` / ${(total / 1048576).toFixed(0)} MB` : ''}`);
  }

  const buffer = await new Blob(chunks).arrayBuffer();

  try {
    const cache = await caches.open('upsr-model-v2');
    await cache.put(url, new Response(buffer.slice(0), {
      headers: { 'Content-Type': 'application/octet-stream' },
    }));
  } catch (_) { /* 缓存失败不影响使用 */ }

  return buffer;
}

async function ensureSession(key) {
  if (sessions[key]) return sessions[key];

  const cfg = MODELS[key];
  const buffer = await fetchModelBuffer(cfg.url, (r, t) => setProgress(r * 0.35, t));

  setProgress(0.36, '初始化推理引擎…');

  const tries = navigator.gpu ? [['webgpu', 'wasm'], ['wasm']] : [['wasm']];
  let lastErr = null;

  for (const eps of tries) {
    try {
      const sess = await ort.InferenceSession.create(buffer, {
        executionProviders: eps,
        graphOptimizationLevel: 'all',
      });
      sessions[key] = {
        sess,
        input: cfg.input,
        output: cfg.output,
        ep: eps[0] === 'webgpu' ? 'WebGPU' : 'WebAssembly',
      };
      mDev.textContent = sessions[key].ep;
      return sessions[key];
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error(`推理引擎初始化失败：${lastErr?.message || '未知错误'}`);
}

/* ---------------- 图像准备 ---------------- */
function prepareCanvases() {
  const maxInput = MODELS[currentModel].maxInput;
  const ow = srcBitmap.width;
  const oh = srcBitmap.height;

  let w = ow;
  let h = oh;
  let scaled = false;

  if (Math.max(w, h) > maxInput) {
    const k = maxInput / Math.max(w, h);
    w = Math.round(w * k);
    h = Math.round(h * k);
    scaled = true;
  }

  cvBefore.width = w;
  cvBefore.height = h;
  cvBefore.getContext('2d').drawImage(srcBitmap, 0, 0, w, h);

  cvAfter.width = w * SCALE;
  cvAfter.height = h * SCALE;
  cvAfter.getContext('2d').clearRect(0, 0, w * SCALE, h * SCALE);

  mSize.textContent = `${ow} × ${oh}${scaled ? ` → ${w} × ${h}` : ''}`;
  mOut.textContent = `${w * SCALE} × ${h * SCALE}`;

  return { w, h, scaled };
}

async function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showErr('请选择图片文件（JPG / PNG / WebP）。');
    return;
  }
  clearErr();
  resultBlob = null;
  dlBtn.disabled = true;

  srcBitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
    .catch(() => createImageBitmap(file));

  prepareCanvases();

  mName.textContent = file.name;
  mTime.textContent = '—';

  dropEl.hidden = true;
  workEl.hidden = false;
  progEl.hidden = true;
  setBusy(false);
}

/* ---------------- 推理 ---------------- */
async function upscale(onProgress) {
  const W = cvBefore.width;
  const H = cvBefore.height;
  const srcData = cvBefore.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, W, H).data;

  const OW = W * SCALE;
  const OH = H * SCALE;
  const outImage = new ImageData(OW, OH);
  const od = outImage.data;

  const tilesX = Math.ceil(W / TILE);
  const tilesY = Math.ceil(H / TILE);
  const total = tilesX * tilesY;
  let done = 0;

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * TILE;
      const y0 = ty * TILE;
      const w = Math.min(TILE, W - x0);
      const h = Math.min(TILE, H - y0);
      const pw = w + MARGIN * 2;
      const ph = h + MARGIN * 2;
      const plane = pw * ph;

      // 组装 NCHW 浮点输入，越界按反射取样
      const input = new Float32Array(3 * plane);
      for (let yy = 0; yy < ph; yy++) {
        const sy = reflect(y0 - MARGIN + yy, H);
        const rowBase = sy * W;
        const dstBase = yy * pw;
        for (let xx = 0; xx < pw; xx++) {
          const sx = reflect(x0 - MARGIN + xx, W);
          const si = (rowBase + sx) * 4;
          input[dstBase + xx] = srcData[si] / 255;
          input[plane + dstBase + xx] = srcData[si + 1] / 255;
          input[2 * plane + dstBase + xx] = srcData[si + 2] / 255;
        }
      }

      const m = sessions[currentModel];
      const feeds = { [m.input]: new ort.Tensor('float32', input, [1, 3, ph, pw]) };
      const results = await m.sess.run(feeds);
      const output = results[m.output].data;

      const ow = pw * SCALE;
      const oh = ph * SCALE;
      const oplane = ow * oh;
      const om = MARGIN * SCALE;   // 输出里要裁掉的边距
      const outW = w * SCALE;
      const outH = h * SCALE;
      const dstX = x0 * SCALE;
      const dstY = y0 * SCALE;

      for (let yy = 0; yy < outH; yy++) {
        const syy = yy + om;
        const srcRow = syy * ow;
        const dstRow = (dstY + yy) * OW + dstX;
        for (let xx = 0; xx < outW; xx++) {
          const sxx = srcRow + xx + om;
          const oi = (dstRow + xx) * 4;
          od[oi] = clamp255(output[sxx] * 255);
          od[oi + 1] = clamp255(output[oplane + sxx] * 255);
          od[oi + 2] = clamp255(output[2 * oplane + sxx] * 255);
          od[oi + 3] = 255;
        }
      }

      done++;
      onProgress?.(0.4 + 0.6 * (done / total), `处理第 ${done} / ${total} 块`);
      await new Promise((r) => setTimeout(r, 0)); // 让出主线程刷新进度
    }
  }

  // 原图有透明通道时按比例放大保留
  let hasAlpha = false;
  for (let i = 3; i < srcData.length; i += 4) {
    if (srcData[i] !== 255) { hasAlpha = true; break; }
  }

  if (hasAlpha) {
    const mask = document.createElement('canvas');
    mask.width = W;
    mask.height = H;
    mask.getContext('2d').drawImage(cvBefore, 0, 0);

    const big = document.createElement('canvas');
    big.width = OW;
    big.height = OH;
    const bctx = big.getContext('2d');
    bctx.imageSmoothingQuality = 'high';
    bctx.drawImage(mask, 0, 0, OW, OH);
    const alphaData = bctx.getImageData(0, 0, OW, OH).data;
    for (let i = 3; i < od.length; i += 4) od[i] = alphaData[i];
  }

  return outImage;
}

/* ---------------- 输出 ---------------- */
function paintResult(imageData) {
  cvAfter.width = imageData.width;
  cvAfter.height = imageData.height;
  cvAfter.getContext('2d').putImageData(imageData, 0, 0);
}

async function reduceTo(imageData, factor) {
  const tmp = document.createElement('canvas');
  tmp.width = imageData.width;
  tmp.height = imageData.height;
  tmp.getContext('2d').putImageData(imageData, 0, 0);

  const ow = Math.round(imageData.width / factor);
  const oh = Math.round(imageData.height / factor);
  const out = document.createElement('canvas');
  out.width = ow;
  out.height = oh;
  const octx = out.getContext('2d');
  octx.imageSmoothingEnabled = true;
  octx.imageSmoothingQuality = 'high';
  octx.drawImage(tmp, 0, 0, ow, oh);
  return out;
}

/* ---------------- 主流程 ---------------- */
async function run() {
  if (busy || !cvBefore.width) return;
  clearErr();
  setBusy(true);
  resultBlob = null;

  const t0 = performance.now();

  try {
    await ensureSession(currentModel);
    const img = await upscale((r, t) => setProgress(r, t));
    const factor = scaleSel.value === '2' ? 2 : 1;

    if (factor === 2) {
      const small = await reduceTo(img, 2);
      cvAfter.width = small.width;
      cvAfter.height = small.height;
      cvAfter.getContext('2d').drawImage(small, 0, 0);
      mOut.textContent = `${small.width} × ${small.height}`;
      resultBlob = await new Promise((res) => small.toBlob(res, 'image/png'));
    } else {
      paintResult(img);
      mOut.textContent = `${img.width} × ${img.height}`;
      resultBlob = await new Promise((res) => cvAfter.toBlob(res, 'image/png'));
    }

    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    mTime.textContent = `${secs} 秒`;
    setProgress(1, `完成，用时 ${secs} 秒`);
    dlBtn.disabled = false;
  } catch (e) {
    console.error(e);
    showErr(e?.message || String(e));
    progEl.hidden = true;
  } finally {
    setBusy(false);
  }
}

function resetAll() {
  srcBitmap = null;
  resultBlob = null;
  fileEl.value = '';
  workEl.hidden = true;
  dropEl.hidden = false;
  progEl.hidden = true;
  clearErr();
  cvAfter.width = 0;
  cvAfter.height = 0;
  mName.textContent = '—';
  mSize.textContent = '—';
  mOut.textContent = '—';
  mTime.textContent = '—';
}

/* ---------------- 事件 ---------------- */
dropEl.addEventListener('click', () => fileEl.click());
dropEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileEl.click(); }
});

fileEl.addEventListener('change', () => {
  const f = fileEl.files?.[0];
  if (f) loadFile(f).catch((e) => showErr(e.message));
});

['dragenter', 'dragover'].forEach((ev) =>
  dropEl.addEventListener(ev, (e) => { e.preventDefault(); dropEl.classList.add('over'); })
);
['dragleave', 'drop'].forEach((ev) =>
  dropEl.addEventListener(ev, (e) => { e.preventDefault(); dropEl.classList.remove('over'); })
);

dropEl.addEventListener('drop', (e) => {
  e.preventDefault();
  e.stopPropagation();
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f).catch((err) => showErr(err.message));
});

// 拖到页面任意位置都能换图
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f).catch((err) => showErr(err.message));
});

sliderEl.addEventListener('input', () => {
  const v = Number(sliderEl.value);
  cvAfter.style.clipPath = `inset(0 0 0 ${v}%)`;
  handleEl.style.left = `${v}%`;
});

modelSel.addEventListener('change', () => {
  currentModel = modelSel.value;
  updateModelNote();
  if (srcBitmap) prepareCanvases();
});

runBtn.addEventListener('click', run);
againBtn.addEventListener('click', resetAll);

dlBtn.addEventListener('click', () => {
  if (!resultBlob) return;
  const url = URL.createObjectURL(resultBlob);
  const a = document.createElement('a');
  const base = (mName.textContent || 'image').replace(/\.[^.]+$/, '');
  a.href = url;
  a.download = `${base}_x${scaleSel.value}.png`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
});

updateModelNote();
