/* ══════════════════════════════════════════════════════════
   NoiseLab — Core Processing Engine
   ══════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────
let sourceImageData = null;   // Original grayscale ImageData
let sourceW = 0, sourceH = 0;
let processedCanvases = {};   // { key: HTMLCanvasElement }
let processedLabels = {};     // { key: string }
let processedMetrics = {};    // { key: {psnr, mse} }
let currentParams = {};       // { key: string (param readout) }

// ── Theme is strictly hardcoded to dark ──

// ── Drag & Drop ──────────────────────────────────────────
const dz = document.getElementById('dropzone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// ── File Handler ─────────────────────────────────────────
function handleFile(file) {
  // Validate type
  if (!file.type.startsWith('image/')) {
    showToast('Invalid file type. Please upload an image.');
    return;
  }
  // Validate size (20MB)
  if (file.size > 20 * 1024 * 1024) {
    showToast('File too large. Max size is 20MB.');
    return;
  }

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;

  img.onload = () => {
    // Load into proc canvas
    const pc = document.getElementById('proc-canvas');
    pc.width  = img.naturalWidth;
    pc.height = img.naturalHeight;
    const ctx = pc.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const raw = ctx.getImageData(0, 0, pc.width, pc.height);
    sourceImageData = toGrayscale(raw);
    sourceW = pc.width;
    sourceH = pc.height;

    // Show background in dz-bg if exists
    const bg = document.getElementById('dz-bg');
    if (bg) bg.style.backgroundImage = `url(${url})`;
    const title = document.getElementById('dz-title');
    if (title) title.textContent = file.name;
    const sub = document.getElementById('dz-sub');
    if (sub) sub.textContent = 'Upload complete';

    // Show panels
    document.getElementById('controls-section').classList.remove('hidden');

    // Reset results
    resetResults();
  };
}



// ── Example Image ────────────────────────────────────────
function loadExampleImage(url, filename) {
  if (!url) return;
  showSpinner('Loading', `Fetching ${filename || 'example'}...`);
  
  const img = new Image();
  if (url.startsWith('http')) {
    img.crossOrigin = "anonymous";
  }
  img.src = url;
  
  img.onload = () => {
    try {
      const pc = document.getElementById('proc-canvas');
      pc.width  = img.naturalWidth;
      pc.height = img.naturalHeight;
      const ctx = pc.getContext('2d');
      ctx.drawImage(img, 0, 0);
      
      const raw = ctx.getImageData(0, 0, pc.width, pc.height);
      sourceImageData = toGrayscale(raw);
      sourceW = pc.width;
      sourceH = pc.height;

      const bg = document.getElementById('dz-bg');
      if (bg) bg.style.backgroundImage = `url('${url}')`;
      const title = document.getElementById('dz-title');
      if (title) title.textContent = filename || 'Example Image';
      const sub = document.getElementById('dz-sub');
      if (sub) sub.textContent = 'Upload complete';

      document.getElementById('controls-section').classList.remove('hidden');
      resetResults();
      hideSpinner();
    } catch (e) {
      hideSpinner();
      showToast('Canvas tainted. Open via localhost instead of file://');
      console.error(e);
    }
  };
  
  img.onerror = () => {
    hideSpinner();
    showToast(`Failed to load image from network`);
  };
}

// ── Convert to Grayscale ─────────────────────────────────
function toGrayscale(imgData) {
  const src  = imgData.data;
  const out  = new Uint8ClampedArray(src.length);
  for (let i = 0; i < src.length; i += 4) {
    // Luminosity formula
    const g = Math.round(0.299 * src[i] + 0.587 * src[i+1] + 0.114 * src[i+2]);
    out[i] = out[i+1] = out[i+2] = g;
    out[i+3] = 255;
  }
  return new ImageData(out, imgData.width, imgData.height);
}

// ── PSNR Calculation ─────────────────────────────────────
// PSNR = 20·log10(MAX) - 10·log10(MSE)
function calcPSNR_MSE(orig, noisy) {
  const a = orig.data, b = noisy.data;
  let mse = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 4) {
    const diff = a[i] - b[i];      // Only need R channel (grayscale)
    mse += diff * diff;
    count++;
  }
  mse /= count;
  if (mse === 0) return { psnr: Infinity, mse: 0 };
  const psnr = 20 * Math.log10(255) - 10 * Math.log10(mse);
  return { psnr: +psnr.toFixed(2), mse: +mse.toFixed(4) };
}

// ── Gaussian Noise ────────────────────────────────────────
// Box-Muller transform to generate Gaussian random numbers
function addGaussianNoise(imgData, sigma) {
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);
  for (let i = 0; i < src.length; i += 4) {
    // Box-Muller
    const u1 = Math.random(), u2 = Math.random();
    const z  = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const noisy = src[i] + z * sigma;
    const clamped = Math.max(0, Math.min(255, Math.round(noisy)));
    out[i] = out[i+1] = out[i+2] = clamped;
    out[i+3] = 255;
  }
  return new ImageData(out, imgData.width, imgData.height);
}

// ── Salt & Pepper Noise ───────────────────────────────────
function addSaltPepperNoise(imgData, prob) {
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);
  const halfProb = prob / 2;
  for (let i = 0; i < src.length; i += 4) {
    const r = Math.random();
    let v;
    if      (r < halfProb)            v = 0;    // Pepper (black)
    else if (r < prob)                v = 255;  // Salt (white)
    else                              v = src[i]; // Original
    out[i] = out[i+1] = out[i+2] = v;
    out[i+3] = 255;
  }
  return new ImageData(out, imgData.width, imgData.height);
}

// ── Mean Filter ──────────────────────────────────────────
// Averages pixel values in an N×N neighbourhood
function meanFilter(imgData, k) {
  const { width: w, height: h } = imgData;
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);
  const half = Math.floor(k / 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let ky = -half; ky <= half; ky++) {
        for (let kx = -half; kx <= half; kx++) {
          const ny = Math.min(Math.max(y+ky, 0), h-1);
          const nx = Math.min(Math.max(x+kx, 0), w-1);
          sum += src[(ny*w + nx)*4];
          count++;
        }
      }
      const v = Math.round(sum / count);
      const idx = (y*w + x)*4;
      out[idx] = out[idx+1] = out[idx+2] = v;
      out[idx+3] = 255;
    }
  }
  return new ImageData(out, w, h);
}

// ── Median Filter ────────────────────────────────────────
// Replaces each pixel with the median of its neighbourhood
function medianFilter(imgData, k) {
  const { width: w, height: h } = imgData;
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);
  const half = Math.floor(k / 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const vals = [];
      for (let ky = -half; ky <= half; ky++) {
        for (let kx = -half; kx <= half; kx++) {
          const ny = Math.min(Math.max(y+ky, 0), h-1);
          const nx = Math.min(Math.max(x+kx, 0), w-1);
          vals.push(src[(ny*w + nx)*4]);
        }
      }
      vals.sort((a,b) => a-b);
      const v = vals[Math.floor(vals.length/2)];
      const idx = (y*w + x)*4;
      out[idx] = out[idx+1] = out[idx+2] = v;
      out[idx+3] = 255;
    }
  }
  return new ImageData(out, w, h);
}

// ── Gaussian Kernel ──────────────────────────────────────
function makeGaussianKernel(k, sigma) {
  const half   = Math.floor(k / 2);
  const kernel = [];
  let sum = 0;
  for (let y = -half; y <= half; y++) {
    for (let x = -half; x <= half; x++) {
      const v = Math.exp(-(x*x + y*y) / (2 * sigma * sigma));
      kernel.push(v);
      sum += v;
    }
  }
  return kernel.map(v => v / sum); // Normalize
}

// ── Gaussian Filter ──────────────────────────────────────
function gaussianFilter(imgData, k) {
  const { width: w, height: h } = imgData;
  const src    = imgData.data;
  const out    = new Uint8ClampedArray(src.length);
  const half   = Math.floor(k / 2);
  // Sigma derived from kernel size using standard formula
  const sigma  = 0.3 * ((k - 1) * 0.5 - 1) + 0.8;
  const kernel = makeGaussianKernel(k, sigma);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0, ki = 0;
      for (let ky = -half; ky <= half; ky++) {
        for (let kx = -half; kx <= half; kx++) {
          const ny = Math.min(Math.max(y+ky, 0), h-1);
          const nx = Math.min(Math.max(x+kx, 0), w-1);
          acc += src[(ny*w + nx)*4] * kernel[ki++];
        }
      }
      const v = Math.round(acc);
      const idx = (y*w + x)*4;
      out[idx] = out[idx+1] = out[idx+2] = v;
      out[idx+3] = 255;
    }
  }
  return new ImageData(out, w, h);
}

// ── ImageData → Canvas ────────────────────────────────────
function imageDataToCanvas(imgData) {
  const c   = document.createElement('canvas');
  c.width   = imgData.width;
  c.height  = imgData.height;
  c.getContext('2d').putImageData(imgData, 0, 0);
  return c;
}

// ── Process Image ────────────────────────────────────────
async function processImage() {
  if (!sourceImageData) return;
  const sigma  = parseInt(document.getElementById('sigma-slider').value);
  const spProb = parseInt(document.getElementById('sp-slider').value) / 100;
  const k      = parseInt(document.getElementById('kernel-size').value);

  const sel = document.getElementById('algo-select').value;
  const doGn = sel === 'all' || sel === 'gn';
  const doSpn = sel === 'all' || sel === 'spn';
  const doMf = sel === 'all' || sel === 'mf';
  const doMedf = sel === 'all' || sel === 'medf';
  const doGf = sel === 'all' || sel === 'gf';

  showSpinner('Running Analysis', 'Applying noise & filters...');

  // Small delay to let spinner render
  await new Promise(r => setTimeout(r, 30));

  try {
    let gaussNoise, spNoise, meanOut, medOut, gaussOut;

    /* ── Step 1: Noisy images ─────────────────────────── */
    if (doGn) {
      updateSpinner('Adding Gaussian Noise...');
      await tick();
      gaussNoise = addGaussianNoise(sourceImageData, sigma);
    }
    if (doSpn) {
      updateSpinner('Adding Salt & Pepper Noise...');
      await tick();
      spNoise = addSaltPepperNoise(sourceImageData, spProb);
    }

    /* ── Step 2: Apply filters ────────────────────────── */
    if (doMf) {
      updateSpinner('Applying Mean Filter...');
      await tick();
      meanOut = meanFilter(gaussNoise || sourceImageData, k);
    }
    if (doMedf) {
      updateSpinner('Applying Median Filter...');
      await tick();
      medOut = medianFilter(spNoise || sourceImageData, k);
    }
    if (doGf) {
      updateSpinner('Applying Gaussian Filter...');
      await tick();
      gaussOut = gaussianFilter(gaussNoise || sourceImageData, k);
    }

    /* ── Step 3: PSNR/MSE ───────────────────────────── */
    const origRef = sourceImageData;
    const metrics = {
      original: { psnr: Infinity, mse: 0 }
    };
    if (doGn) metrics.gaussNoise = calcPSNR_MSE(origRef, gaussNoise);
    if (doSpn) metrics.spNoise = calcPSNR_MSE(origRef, spNoise);
    if (doMf) metrics.meanOut = calcPSNR_MSE(origRef, meanOut);
    if (doMedf) metrics.medOut = calcPSNR_MSE(origRef, medOut);
    if (doGf) metrics.gaussOut = calcPSNR_MSE(origRef, gaussOut);

    /* ── Step 4: Render ──────────────────────────────── */
    processedCanvases = {
      original: imageDataToCanvas(origRef)
    };
    if (doGn) processedCanvases.gaussNoise = imageDataToCanvas(gaussNoise);
    if (doSpn) processedCanvases.spNoise = imageDataToCanvas(spNoise);
    if (doMf) processedCanvases.meanOut = imageDataToCanvas(meanOut);
    if (doMedf) processedCanvases.medOut = imageDataToCanvas(medOut);
    if (doGf) processedCanvases.gaussOut = imageDataToCanvas(gaussOut);

    processedLabels = {
      original:   'Original (Gray)'
    };
    if (doGn) processedLabels.gaussNoise = 'Gaussian Noise';
    if (doSpn) processedLabels.spNoise = 'Salt & Pepper';
    if (doMf) processedLabels.meanOut = 'Mean Filter';
    if (doMedf) processedLabels.medOut = 'Median Filter';
    if (doGf) processedLabels.gaussOut = 'Gaussian Filter';

    processedMetrics = metrics;
    currentParams = { original: 'Source Image' };
    if (doGn) currentParams.gaussNoise = `σ=${sigma}`;
    if (doSpn) currentParams.spNoise = `${(spProb*100).toFixed(0)}%`;
    if (doMf) currentParams.meanOut = `${k}×${k}`;
    if (doMedf) currentParams.medOut = `${k}×${k}`;
    if (doGf) currentParams.gaussOut = `${k}×${k}`;

    setupCompare();

  } catch(e) {
    showToast('Processing error: ' + e.message);
    console.error(e);
  }

  hideSpinner();
}

// ── Compare Slider ────────────────────────────────────────
function setupCompare() {
  const keys = Object.keys(processedCanvases);
  const labels = processedLabels;
  const selA = document.getElementById('compare-a');
  const selB = document.getElementById('compare-b');
  selA.innerHTML = selB.innerHTML = '';
  keys.forEach(k => {
    selA.innerHTML += `<option value="${k}">${labels[k]}</option>`;
    selB.innerHTML += `<option value="${k}">${labels[k]}</option>`;
  });
  selA.value = keys[0];
  selB.value = keys[1] || keys[0];
  document.getElementById('compare-section').classList.remove('hidden');
  updateCompare();
  initCompareSlider();
}

function updateCompare() {
  const keyA = document.getElementById('compare-a').value;
  const keyB = document.getElementById('compare-b').value;
  const canvA = processedCanvases[keyA];
  const canvB = processedCanvases[keyB];
  if (!canvA || !canvB) return;

  const outA = document.getElementById('compare-canvas-a');
  const outB = document.getElementById('compare-canvas-b');
  [outA, outB].forEach(c => {
    c.width = canvA.width; c.height = canvA.height;
    c.style.cssText = 'width:100%;display:block;';
  });
  outA.getContext('2d').drawImage(canvA, 0, 0);
  outB.getContext('2d').drawImage(canvB, 0, 0);

  document.getElementById('label-a').textContent = processedLabels[keyA];
  document.getElementById('label-b').textContent = processedLabels[keyB];

  const metA = processedMetrics[keyA];
  if(metA && document.getElementById('param-a')) {
    document.getElementById('param-a').textContent = currentParams[keyA] || 'N/A';
    document.getElementById('psnr-a').textContent = isFinite(metA.psnr) ? metA.psnr.toFixed(2) + ' dB' : '∞ dB';
    document.getElementById('mse-a').textContent = isFinite(metA.mse) ? metA.mse.toFixed(2) : '0';
  }
  
  const metB = processedMetrics[keyB];
  if(metB && document.getElementById('param-b')) {
    document.getElementById('param-b').textContent = currentParams[keyB] || 'N/A';
    document.getElementById('psnr-b').textContent = isFinite(metB.psnr) ? metB.psnr.toFixed(2) + ' dB' : '∞ dB';
    document.getElementById('mse-b').textContent = isFinite(metB.mse) ? metB.mse.toFixed(2) : '0';
  }

  // Reset divider to 50%
  setComparePos(0.5);
}

function setComparePos(frac) {
  frac = Math.max(0, Math.min(1, frac));
  const pct = (frac * 100).toFixed(1) + '%';
  document.getElementById('compare-canvas-b').style.clipPath = `inset(0 ${100-frac*100}% 0 0)`;
  document.getElementById('compare-divider').style.left = pct;
  document.getElementById('compare-handle').style.left  = pct;
}

function initCompareSlider() {
  const inner = document.getElementById('compare-inner');
  let dragging = false;
  const move = e => {
    if (!dragging) return;
    const rect = inner.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    setComparePos(x / rect.width);
  };
  inner.addEventListener('mousedown',  () => { dragging = true; });
  inner.addEventListener('touchstart', () => { dragging = true; }, { passive: true });
  window.addEventListener('mouseup',   () => { dragging = false; });
  window.addEventListener('touchend',  () => { dragging = false; });
  window.addEventListener('mousemove', move);
  window.addEventListener('touchmove', move, { passive: true });
}

// ── Download Canvas ──────────────────────────────────────
function downloadCanvas(ab) {
  const key = document.getElementById('compare-' + ab).value;
  const c = processedCanvases[key];
  if (!c) return;
  const label = processedLabels[key] || 'image';
  const name = label.replace(/\s/g,'-').toLowerCase();

  const a = document.createElement('a');
  a.href = c.toDataURL('image/png');
  a.download = `noiselab-${name}.png`;
  a.click();
}

// ── Spinner helpers ──────────────────────────────────────
function showSpinner(label, sub) {
  document.getElementById('spinner-label').textContent = label;
  document.getElementById('spinner-sub').textContent   = sub;
  document.getElementById('spinner').classList.add('active');
}
function updateSpinner(sub) { document.getElementById('spinner-sub').textContent = sub; }
function hideSpinner()  { document.getElementById('spinner').classList.remove('active'); }

// ── Toast ─────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  document.getElementById('toast-msg').textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4000);
}

// ── Reset ─────────────────────────────────────────────────
function resetAll() {
  sourceImageData = null;
  processedCanvases = {};
  document.getElementById('controls-section').classList.add('hidden');
  document.getElementById('compare-section').classList.add('hidden');
  document.getElementById('file-input').value = '';
}
function resetResults() {
  const compSec = document.getElementById('compare-section');
  if (compSec) compSec.classList.add('hidden');
}

// ── Helpers ───────────────────────────────────────────────
const formatBytes = b =>
  b < 1024 ? b + ' B' :
  b < 1048576 ? (b/1024).toFixed(1) + ' KB' :
  (b/1048576).toFixed(2) + ' MB';

const tick = () => new Promise(r => setTimeout(r, 10));
