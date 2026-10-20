/**
 * ApexSR Studio - Peak UI Client Logic
 * Handles interactive split-slider, magnifier loupe, zoom, clipboard upload,
 * neural inference requests, metrics telemetry, and comparison card export.
 */

(function () {
  'use strict';

  // State Management
  const state = {
    currentImage: null, // raw image file or sample id
    currentMode: 'split', // 'split', 'side', 'bicubic', 'loupe'
    zoomLevel: 1.0,
    splitPosition: 50, // in percent
    isDraggingSplit: false,
    isProcessing: false,
    activeProfile: 'balanced',
    format: 'png',
    data: null, // Last inference result { input_b64, srgan_b64, bicubic_b64, metrics }
    samples: []
  };

  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const activeViewport = document.getElementById('activeViewport');
  const viewModeBar = document.getElementById('viewModeBar');
  const stageContainer = document.getElementById('stageContainer');
  const splitSliderWrapper = document.getElementById('splitSliderWrapper');
  const sliderDivider = document.getElementById('sliderDivider');
  const sideWrapper = document.getElementById('sideWrapper');
  const bicubicWrapper = document.getElementById('bicubicWrapper');
  const loupeLens = document.getElementById('loupeLens');
  const loupeCanvas = document.getElementById('loupeCanvas');
  const loupeCtx = loupeCanvas.getContext('2d');
  const processingOverlay = document.getElementById('processingOverlay');
  const samplesGrid = document.getElementById('samplesGrid');

  // Images
  const originalImg = document.getElementById('originalImg');
  const refinedImg = document.getElementById('refinedImg');
  const sideOriginalImg = document.getElementById('sideOriginalImg');
  const sideRefinedImg = document.getElementById('sideRefinedImg');
  const bicubicImg = document.getElementById('bicubicImg');
  const bicubicRefinedImg = document.getElementById('bicubicRefinedImg');

  // Badges & Labels
  const labelInputDim = document.getElementById('labelInputDim');
  const labelOutputDim = document.getElementById('labelOutputDim');
  const mScale = document.getElementById('mScale');
  const mLatency = document.getElementById('mLatency');
  const mSharpness = document.getElementById('mSharpness');
  const mPsnr = document.getElementById('mPsnr');
  const zoomLevelText = document.getElementById('zoomLevelText');

  // Controls
  const sharpnessSlider = document.getElementById('sharpnessSlider');
  const sharpnessVal = document.getElementById('sharpnessVal');
  const denoiseSlider = document.getElementById('denoiseSlider');
  const denoiseVal = document.getElementById('denoiseVal');
  const contrastToggle = document.getElementById('contrastToggle');
  const maxDimSelect = document.getElementById('maxDimSelect');
  const reRefineBtn = document.getElementById('reRefineBtn');
  const resetSlidersBtn = document.getElementById('resetSlidersBtn');
  const downloadRefinedBtn = document.getElementById('downloadRefinedBtn');
  const downloadCardBtn = document.getElementById('downloadCardBtn');
  const newImageBtn = document.getElementById('newImageBtn');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const zoomFitBtn = document.getElementById('zoomFitBtn');
  const shortcutsBtn = document.getElementById('shortcutsBtn');
  const shortcutsModal = document.getElementById('shortcutsModal');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const exportCanvas = document.getElementById('exportCanvas');

  // Initialize
  function init() {
    setupEventListeners();
    setupDropZone();
    setupClipboardPaste();
    setupSplitSlider();
    setupLoupe();
    fetchBackendStatus();
    fetchSamples();
  }

  function getApiBase() {
    return (window.APEX_API_BASE || localStorage.getItem('apex_api_base') || '').replace(/\/$/, '');
  }

  // Check Backend Status
  async function fetchBackendStatus() {
    const apiBase = getApiBase();
    const pill = document.getElementById('backendStatusPill');
    try {
      const res = await fetch(`${apiBase}/api/status`);
      if (res.ok) {
        const data = await res.json();
        if (pill && data.model_loaded) {
          pill.querySelector('.status-label').textContent = `SRGAN 4x • ${data.backend.toUpperCase()} Ready`;
          return;
        }
      }
    } catch (e) {
      console.warn("Could not reach backend status endpoint", e);
    }
    if (pill) {
      pill.querySelector('.status-label').textContent = `Set Backend API URL`;
      pill.style.cursor = 'pointer';
      pill.title = 'Click to connect a remote backend URL (e.g. Hugging Face Space)';
      pill.onclick = () => {
        const current = getApiBase();
        const nextUrl = prompt('Enter your SRGAN Backend API URL (e.g. Hugging Face Space or Render URL):', current);
        if (nextUrl !== null) {
          localStorage.setItem('apex_api_base', nextUrl.trim());
          window.location.reload();
        }
      };
    }
  }

  // Fetch Samples
  async function fetchSamples() {
    try {
      const res = await fetch(`${getApiBase()}/api/samples`);
      if (!res.ok) return;
      const data = await res.json();
      state.samples = data.samples || [];
      renderSamples();
    } catch (err) {
      console.warn('Failed to fetch samples', err);
    }
  }

  function renderSamples() {
    if (!samplesGrid) return;
    if (state.samples.length === 0) {
      samplesGrid.innerHTML = '<div style="grid-column: span 4; text-align: center; color: var(--text-dim); font-size: 0.8rem; padding: 1rem;">No preset samples found. Upload any image above!</div>';
      return;
    }

    samplesGrid.innerHTML = state.samples.map(s => `
      <div class="sample-card" data-sample-id="${s.id}">
        <div class="sample-img-box">
          <img src="${s.url}" alt="${s.title}" loading="lazy">
          <span class="sample-overlay-tag">${s.category}</span>
        </div>
        <div class="sample-meta">
          <div class="sample-name">${s.title}</div>
          <div class="sample-desc">${s.description}</div>
        </div>
      </div>
    `).join('');

    // Sample click events
    samplesGrid.querySelectorAll('.sample-card').forEach(card => {
      card.addEventListener('click', () => {
        const sampleId = card.getAttribute('data-sample-id');
        loadSample(sampleId);
      });
    });
  }

  // Load a sample
  function loadSample(sampleId) {
    state.currentImage = { type: 'sample', id: sampleId };
    executeRefine();
  }

  // Setup Event Listeners
  function setupEventListeners() {
    // File Input
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });

    // New Image Button
    newImageBtn.addEventListener('click', () => {
      dropZone.classList.remove('hidden');
      activeViewport.classList.add('hidden');
      viewModeBar.classList.add('hidden');
      reRefineBtn.disabled = true;
      downloadRefinedBtn.disabled = true;
      downloadCardBtn.disabled = true;
      fileInput.value = '';
    });

    // Mode Buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-mode');
        switchViewMode(mode);
      });
    });

    // Profiles
    document.querySelectorAll('.profile-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.profile-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const profile = chip.getAttribute('data-profile');
        applyProfile(profile);
      });
    });

    // Sliders
    sharpnessSlider.addEventListener('input', (e) => {
      sharpnessVal.textContent = `${e.target.value}%`;
    });

    denoiseSlider.addEventListener('input', (e) => {
      denoiseVal.textContent = `${e.target.value}%`;
    });

    // Reset Sliders
    resetSlidersBtn.addEventListener('click', () => {
      applyProfile('balanced');
    });

    // Re-Refine Button
    reRefineBtn.addEventListener('click', () => {
      if (state.currentImage) {
        executeRefine();
      }
    });

    // Format buttons
    document.querySelectorAll('.format-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.format-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.format = btn.getAttribute('data-fmt');
      });
    });

    // Download Buttons
    downloadRefinedBtn.addEventListener('click', downloadRefinedImage);
    downloadCardBtn.addEventListener('click', exportComparisonCard);

    // Zoom Controls
    zoomInBtn.addEventListener('click', () => adjustZoom(0.25));
    zoomOutBtn.addEventListener('click', () => adjustZoom(-0.25));
    zoomFitBtn.addEventListener('click', () => setZoom(1.0));

    // Fullscreen Toggle
    fullscreenBtn.addEventListener('click', toggleFullscreen);

    // Shortcuts Modal
    shortcutsBtn.addEventListener('click', () => shortcutsModal.classList.remove('hidden'));
    closeModalBtn.addEventListener('click', () => shortcutsModal.classList.add('hidden'));
    shortcutsModal.addEventListener('click', (e) => {
      if (e.target === shortcutsModal) shortcutsModal.classList.add('hidden');
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.key === '?' || (e.shiftKey && e.key === '/')) {
        shortcutsModal.classList.toggle('hidden');
      } else if (e.key === 'Escape') {
        shortcutsModal.classList.add('hidden');
        if (document.fullscreenElement) {
          document.exitFullscreen();
        }
      } else if (e.key === 'f' || e.key === 'F') {
        if (!['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
          toggleFullscreen();
        }
      } else if (e.key === 's' || e.key === 'S') {
        if (!['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
          switchViewMode('side');
        }
      } else if (e.key === 'b' || e.key === 'B') {
        if (!['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
          switchViewMode('bicubic');
        }
      } else if (e.key === 'l' || e.key === 'L') {
        if (!['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
          switchViewMode('loupe');
        }
      } else if (e.key === 'ArrowLeft') {
        setSplitPosition(Math.max(0, state.splitPosition - 5));
      } else if (e.key === 'ArrowRight') {
        setSplitPosition(Math.min(100, state.splitPosition + 5));
      }
    });
  }

  // Profiles Definition
  function applyProfile(profile) {
    state.activeProfile = profile;
    switch (profile) {
      case 'crisp':
        sharpnessSlider.value = 65;
        denoiseSlider.value = 0;
        contrastToggle.checked = false;
        break;
      case 'portrait':
        sharpnessSlider.value = 10;
        denoiseSlider.value = 35;
        contrastToggle.checked = false;
        break;
      case 'vivid':
        sharpnessSlider.value = 35;
        denoiseSlider.value = 0;
        contrastToggle.checked = true;
        break;
      case 'balanced':
      default:
        sharpnessSlider.value = 25;
        denoiseSlider.value = 0;
        contrastToggle.checked = false;
        break;
    }
    sharpnessVal.textContent = `${sharpnessSlider.value}%`;
    denoiseVal.textContent = `${denoiseSlider.value}%`;

    // Highlight chip
    document.querySelectorAll('.profile-chip').forEach(c => {
      c.classList.toggle('active', c.getAttribute('data-profile') === profile);
    });

    if (state.data) {
      showToast(`Applied "${profile}" profile. Click Re-Process to render.`, 'info');
    }
  }

  // Drag & Drop
  function setupDropZone() {
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.querySelector('.dropzone-card').classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.querySelector('.dropzone-card').classList.remove('drag-over');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });
  }

  // Clipboard Paste Support
  function setupClipboardPaste() {
    window.addEventListener('paste', (e) => {
      if (e.clipboardData && e.clipboardData.items) {
        for (let i = 0; i < e.clipboardData.items.length; i++) {
          const item = e.clipboardData.items[i];
          if (item.type.indexOf('image') !== -1) {
            const blob = item.getAsFile();
            handleFileSelect(blob);
            showToast('Image pasted from clipboard!', 'success');
            break;
          }
        }
      }
    });
  }

  function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
      showToast('Please upload a valid image file (PNG, JPG, WebP)', 'error');
      return;
    }
    state.currentImage = { type: 'file', file: file };
    executeRefine();
  }

  // Refine Inference API Call
  async function executeRefine() {
    if (!state.currentImage || state.isProcessing) return;

    state.isProcessing = true;
    processingOverlay.classList.remove('hidden');
    reRefineBtn.disabled = true;

    try {
      const formData = new FormData();
      if (state.currentImage.type === 'file') {
        formData.append('image', state.currentImage.file);
      } else if (state.currentImage.type === 'sample') {
        formData.append('sample_id', state.currentImage.id);
      }

      formData.append('max_dim', maxDimSelect.value);
      formData.append('sharpness', (parseInt(sharpnessSlider.value, 10) / 100).toFixed(2));
      formData.append('denoise', (parseInt(denoiseSlider.value, 10) / 100).toFixed(2));
      formData.append('contrast_boost', contrastToggle.checked ? 'true' : 'false');
      formData.append('output_format', state.format);

      const response = await fetch(`${getApiBase()}/api/refine`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Super-resolution failed');
      }

      const result = await response.json();
      state.data = result;
      displayResults(result);
      showToast('Masterpiece refined successfully (4x Super-Resolution)!', 'success');

    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message}`, 'error');
    } finally {
      state.isProcessing = false;
      processingOverlay.classList.add('hidden');
      reRefineBtn.disabled = false;
      downloadRefinedBtn.disabled = false;
      downloadCardBtn.disabled = false;
    }
  }

  // Display Super-Resolved Results
  function displayResults(data) {
    dropZone.classList.add('hidden');
    activeViewport.classList.remove('hidden');
    viewModeBar.classList.remove('hidden');

    // Load image sources
    originalImg.src = data.input_b64;
    refinedImg.src = data.srgan_b64;

    sideOriginalImg.src = data.input_b64;
    sideRefinedImg.src = data.srgan_b64;

    bicubicImg.src = data.bicubic_b64;
    bicubicRefinedImg.src = data.srgan_b64;

    // Dimensions labels
    labelInputDim.textContent = `${data.metrics.input_width}×${data.metrics.input_height}`;
    labelOutputDim.textContent = `${data.metrics.output_width}×${data.metrics.output_height}`;

    // Telemetry updates
    mScale.textContent = `${data.metrics.scale_factor} / ${data.metrics.pixel_multiplier}`;
    mLatency.textContent = `${data.metrics.inference_time_ms} ms`;
    mSharpness.textContent = `+${data.metrics.sharpness_gain_pct}%`;
    mPsnr.textContent = `${data.metrics.psnr_vs_bicubic} dB`;

    // Reset split slider to center
    setSplitPosition(50);
    setZoom(1.0);
  }

  // Interactive Split Slider Logic
  function setupSplitSlider() {
    const handleMove = (clientX) => {
      const rect = splitSliderWrapper.getBoundingClientRect();
      let pos = ((clientX - rect.left) / rect.width) * 100;
      pos = Math.max(0, Math.min(100, pos));
      setSplitPosition(pos);
    };

    const onPointerDown = (e) => {
      if (state.currentMode !== 'split') return;
      state.isDraggingSplit = true;
      handleMove(e.clientX || (e.touches && e.touches[0].clientX));
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
    };

    const onPointerMove = (e) => {
      if (!state.isDraggingSplit) return;
      handleMove(e.clientX || (e.touches && e.touches[0].clientX));
    };

    const onPointerUp = () => {
      state.isDraggingSplit = false;
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };

    splitSliderWrapper.addEventListener('pointerdown', onPointerDown);

    // Touch support
    splitSliderWrapper.addEventListener('touchstart', (e) => {
      if (state.currentMode !== 'split') return;
      state.isDraggingSplit = true;
      handleMove(e.touches[0].clientX);
    }, { passive: true });

    splitSliderWrapper.addEventListener('touchmove', (e) => {
      if (!state.isDraggingSplit) return;
      handleMove(e.touches[0].clientX);
    }, { passive: true });

    splitSliderWrapper.addEventListener('touchend', () => {
      state.isDraggingSplit = false;
    });
  }

  function setSplitPosition(percent) {
    state.splitPosition = percent;
    document.documentElement.style.setProperty('--split-pos', `${percent}%`);
  }

  // Switch View Mode
  function switchViewMode(mode) {
    state.currentMode = mode;
    
    // Update toolbar buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
    });

    // Hide all containers
    splitSliderWrapper.classList.add('hidden');
    sideWrapper.classList.add('hidden');
    bicubicWrapper.classList.add('hidden');
    loupeLens.classList.add('hidden');

    if (mode === 'split') {
      splitSliderWrapper.classList.remove('hidden');
    } else if (mode === 'side') {
      sideWrapper.classList.remove('hidden');
    } else if (mode === 'bicubic') {
      bicubicWrapper.classList.remove('hidden');
    } else if (mode === 'loupe') {
      splitSliderWrapper.classList.remove('hidden');
      loupeLens.classList.remove('hidden');
    }
  }

  // Magnifier Loupe Lens
  function setupLoupe() {
    stageContainer.addEventListener('mousemove', (e) => {
      if (state.currentMode !== 'loupe' || !state.data) return;

      const rect = stageContainer.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      loupeLens.style.left = `${x}px`;
      loupeLens.style.top = `${y}px`;

      // Draw zoomed image on canvas
      const img = refinedImg;
      if (!img.complete) return;

      const imgRect = img.getBoundingClientRect();
      const relX = (e.clientX - imgRect.left) / imgRect.width;
      const relY = (e.clientY - imgRect.top) / imgRect.height;

      if (relX < 0 || relX > 1 || relY < 0 || relY > 1) {
        loupeLens.classList.add('hidden');
        return;
      }
      loupeLens.classList.remove('hidden');

      const srcW = img.naturalWidth;
      const srcH = img.naturalHeight;
      const cropW = 60;
      const cropH = 60;
      const cropX = Math.max(0, Math.min(srcW - cropW, relX * srcW - cropW / 2));
      const cropY = Math.max(0, Math.min(srcH - cropH, relY * srcH - cropH / 2));

      loupeCtx.clearRect(0, 0, 220, 220);
      loupeCtx.imageSmoothingEnabled = true;
      loupeCtx.drawImage(img, cropX, cropY, cropW, cropH, 0, 0, 220, 220);
    });

    stageContainer.addEventListener('mouseleave', () => {
      if (state.currentMode === 'loupe') {
        loupeLens.classList.add('hidden');
      }
    });
  }

  // Zoom Handling
  function adjustZoom(delta) {
    const newZoom = Math.max(0.5, Math.min(4.0, state.zoomLevel + delta));
    setZoom(newZoom);
  }

  function setZoom(val) {
    state.zoomLevel = val;
    document.documentElement.style.setProperty('--zoom-scale', val);
    zoomLevelText.textContent = `${Math.round(val * 100)}%`;
  }

  // Fullscreen
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      stageContainer.requestFullscreen().catch(err => {
        // Fallback CSS fullscreen
        stageContainer.classList.toggle('stage-fullscreen');
      });
    } else {
      document.exitFullscreen();
    }
  }

  // Download Refined Image
  function downloadRefinedImage() {
    if (!state.data || !state.data.srgan_b64) return;
    const link = document.createElement('a');
    link.href = state.data.srgan_b64;
    const ext = state.format === 'jpg' ? 'jpg' : (state.format === 'webp' ? 'webp' : 'png');
    link.download = `ApexSR_Refined_4x_${Date.now()}.${ext}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Download started!', 'success');
  }

  // Export Presentation Comparison Card (Side-by-Side Canvas)
  function exportComparisonCard() {
    if (!state.data) return;

    showToast('Generating Presentation Comparison Card...', 'info');

    const lrImg = new Image();
    const srImg = new Image();

    lrImg.crossOrigin = 'anonymous';
    srImg.crossOrigin = 'anonymous';

    let loadedCount = 0;
    const onBothLoaded = () => {
      loadedCount++;
      if (loadedCount < 2) return;

      const targetH = 800;
      const lrW = Math.round((lrImg.width / lrImg.height) * targetH);
      const srW = Math.round((srImg.width / srImg.height) * targetH);

      const headerH = 120;
      const footerH = 80;
      const gap = 16;
      const pad = 24;

      const cardW = pad * 2 + lrW + srW + gap;
      const cardH = pad * 2 + headerH + targetH + footerH;

      exportCanvas.width = cardW;
      exportCanvas.height = cardH;
      const ctx = exportCanvas.getContext('2d');

      // Background obsidian gradient
      const bgGrad = ctx.createLinearGradient(0, 0, cardW, cardH);
      bgGrad.addColorStop(0, '#07090E');
      bgGrad.addColorStop(1, '#0E131F');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, cardW, cardH);

      // Card Header Branding
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 36px Outfit, sans-serif';
      ctx.fillText('ApexSR — Neural Image Refinement', pad, pad + 45);

      ctx.fillStyle = '#22D3EE';
      ctx.font = '600 16px "JetBrains Mono", monospace';
      ctx.fillText(`4X GENERATIVE SUPER-RESOLUTION  •  SRGAN ARCHITECTURE (16 RESIDUAL BLOCKS)`, pad, pad + 80);

      // Draw Images
      const imgY = pad + headerH;
      const lrX = pad;
      const srX = pad + lrW + gap;

      // LR image border & render
      ctx.fillStyle = '#020617';
      ctx.fillRect(lrX, imgY, lrW, targetH);
      ctx.imageSmoothingEnabled = false; // pixelated look for LR
      ctx.drawImage(lrImg, lrX, imgY, lrW, targetH);

      // SR image border & render
      ctx.fillStyle = '#020617';
      ctx.fillRect(srX, imgY, srW, targetH);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(srImg, srX, imgY, srW, targetH);

      // Overlay Labels on images
      // LR Label
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.fillRect(lrX + 16, imgY + 16, 260, 48);
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 16px Outfit, sans-serif';
      ctx.fillText('INPUT LOW-RESOLUTION', lrX + 30, imgY + 38);
      ctx.fillStyle = '#94A3B8';
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillText(`${state.data.metrics.input_width} × ${state.data.metrics.input_height} px`, lrX + 30, imgY + 54);

      // SR Label
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.fillRect(srX + 16, imgY + 16, 290, 48);
      ctx.fillStyle = '#818CF8';
      ctx.font = 'bold 16px Outfit, sans-serif';
      ctx.fillText('✦ SRGAN 4X MASTERPIECE', srX + 30, imgY + 38);
      ctx.fillStyle = '#22D3EE';
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillText(`${state.data.metrics.output_width} × ${state.data.metrics.output_height} px (16x Pixels)`, srX + 30, imgY + 54);

      // Footer Telemetry
      const footerY = imgY + targetH + 35;
      ctx.fillStyle = '#64748B';
      ctx.font = '500 14px "JetBrains Mono", monospace';
      ctx.fillText(`Sharpness Gain: +${state.data.metrics.sharpness_gain_pct}%  |  PSNR: ${state.data.metrics.psnr_vs_bicubic} dB  |  Latency: ${state.data.metrics.inference_time_ms} ms`, pad, footerY);

      ctx.fillStyle = '#475569';
      ctx.font = '500 13px Outfit, sans-serif';
      ctx.fillText(`Generated by ApexSR Studio on ${new Date().toLocaleDateString()}`, cardW - pad - 260, footerY);

      // Trigger download
      const dataUrl = exportCanvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = `ApexSR_Comparison_Card_${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      showToast('Comparison Card exported!', 'success');
    };

    lrImg.onload = onBothLoaded;
    srImg.onload = onBothLoaded;

    lrImg.src = state.data.input_b64;
    srImg.src = state.data.srgan_b64;
  }

  // Toast Notification System
  function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'success' ? 'check-circle' : (type === 'error' ? 'alert-triangle' : 'info');
    toast.innerHTML = `
      <i data-lucide="${icon}" style="width: 16px; height: 16px; flex-shrink: 0;"></i>
      <span>${message}</span>
    `;
    container.appendChild(toast);

    if (window.lucide) {
      window.lucide.createIcons({ root: toast });
    }

    setTimeout(() => {
      toast.style.transition = 'all 0.3s ease';
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // Start app
  window.addEventListener('DOMContentLoaded', init);

})();
