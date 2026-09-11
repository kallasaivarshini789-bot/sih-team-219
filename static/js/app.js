/**
 * Satellite Cloud Removal & Analysis System - Frontend Application Logic
 * Smart India Hackathon (SIH 2026) | Problem Statement ID: 26209
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropzoneIdle = document.getElementById('dropzone-idle');
    const dropzonePreview = document.getElementById('dropzone-preview');
    const previewThumb = document.getElementById('preview-thumb');
    const previewFilename = document.getElementById('preview-filename');
    const changeFileBtn = document.getElementById('change-file-btn');
    const enhanceBtn = document.getElementById('enhance-btn');
    const processingIndicator = document.getElementById('processing-indicator');

    // Telemetry KPIs
    const valProcTime = document.getElementById('val-proc-time');
    const valInputSize = document.getElementById('val-input-size');
    const valOutputSize = document.getElementById('val-output-size');
    const valContrastBoost = document.getElementById('val-contrast-boost');
    const valHazeReduction = document.getElementById('val-haze-reduction');
    const valCloudCoverage = document.getElementById('val-cloud-coverage');
    const coverageBarFill = document.getElementById('coverage-bar-fill');
    const coverageNote = document.getElementById('coverage-note');

    // Optical Comparison Viewer & Magnifying Lens
    const compViewport = document.getElementById('comparison-viewport');
    const imgCloudy = document.getElementById('img-cloudy');
    const imgMask = document.getElementById('img-mask');
    const cloudyWrap = document.getElementById('cloudy-clipped-wrap');
    const imgCleared = document.getElementById('img-cleared');
    const sliderDivider = document.getElementById('slider-divider');
    const toggleCloudMask = document.getElementById('toggle-cloud-mask');
    const modeTabs = document.querySelectorAll('.mode-tab');
    const exportPdfBtn = document.getElementById('export-pdf-btn');
    const downloadClearedBtn = document.getElementById('download-cleared-btn');

    // Magnifying Lens Elements
    const toggleMagnifierBtn = document.getElementById('toggle-magnifier-btn');
    const loupeLens = document.getElementById('loupe-lens');
    const loupeImg = document.getElementById('loupe-img');
    const loupeZoomTxt = document.getElementById('loupe-zoom-txt');
    const loupeCoordTxt = document.getElementById('loupe-coord-txt');

    // Sector Analysis Suite
    const sectorTabBtns = document.querySelectorAll('.sector-tab-btn');
    const analysisResultMap = document.getElementById('analysis-result-map');
    const sectorSpinner = document.getElementById('sector-spinner');
    const moduleDisplayTitle = document.getElementById('module-display-title');
    const moduleDisplayDesc = document.getElementById('module-display-desc');
    const sectorStatsList = document.getElementById('sector-stats-list');
    const exportAnalysisBtn = document.getElementById('export-analysis-btn');

    // Dashboard Modal
    const openDashboardBtn = document.getElementById('open-dashboard-btn');
    const closeDashboardBtn = document.getElementById('close-dashboard-btn');
    const dashboardModal = document.getElementById('dashboard-modal');

    // --- State Variables ---
    let selectedFile = null;
    let currentClearedFilename = null;
    let currentSceneData = null;
    let activeSectorModule = 'ndvi';
    let isDraggingSlider = false;
    let currentViewMode = 'split';
    let isMagnifierEnabled = false;
    let currentZoomLevel = 2.5;
    let isDiffCRMode = false;   // false = Fast CV mode, true = DiffCR AI mode

    // Chart instances
    let histogramChart = null;
    let sectorDonutChart = null;
    let radarFingerprintChart = null;

    // Fingerprint values (0 to 100)
    const fingerprintScores = {
        visibility: 96,
        vegetation: 25,
        hydrological: 15,
        urban: 10
    };

    // --- 1. Initialize Chart.js Graphs ---
    function initHistogramChart() {
        const ctx = document.getElementById('histogramChart').getContext('2d');
        const defaultLabels = ['0-15', '16-31', '32-47', '48-63', '64-79', '80-95', '96-111', '112-127', '128-143', '144-159', '160-175', '176-191', '192-207', '208-223', '224-239', '240-255'];
        const defaultBefore = [1.2, 2.4, 3.1, 4.2, 5.8, 8.4, 12.1, 16.5, 24.3, 11.2, 5.4, 2.1, 1.2, 0.8, 0.5, 0.2];
        const defaultAfter = [0.4, 1.1, 2.2, 3.5, 6.2, 9.8, 14.2, 18.1, 15.6, 12.3, 8.4, 5.1, 3.2, 1.9, 0.9, 0.4];

        histogramChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: defaultLabels,
                datasets: [
                    {
                        label: 'Before (Cloudy)',
                        data: defaultBefore,
                        borderColor: '#64748b',
                        backgroundColor: 'rgba(100, 116, 139, 0.15)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        fill: true
                    },
                    {
                        label: 'After (Clear)',
                        data: defaultAfter,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.2)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#0f172a',
                        titleColor: '#f8fafc',
                        bodyColor: '#38bdf8',
                        borderColor: 'rgba(56, 189, 248, 0.3)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 8 }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#64748b', font: { size: 9 }, callback: v => v + '%' }
                    }
                }
            }
        });
    }

    function initRadarFingerprintChart() {
        const ctx = document.getElementById('radarFingerprintChart').getContext('2d');
        radarFingerprintChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Visibility Score', 'Vegetation Index', 'Hydrological Index', 'Urban Footprint'],
                datasets: [{
                    label: 'Regional Metric Fingerprint',
                    data: [
                        fingerprintScores.visibility,
                        fingerprintScores.vegetation,
                        fingerprintScores.hydrological,
                        fingerprintScores.urban
                    ],
                    backgroundColor: 'rgba(6, 182, 212, 0.22)',
                    borderColor: '#06b6d4',
                    borderWidth: 2.5,
                    pointBackgroundColor: '#38bdf8',
                    pointBorderColor: '#ffffff',
                    pointHoverBackgroundColor: '#ffffff',
                    pointHoverBorderColor: '#06b6d4',
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(56, 189, 248, 0.15)' },
                        grid: { color: 'rgba(56, 189, 248, 0.12)' },
                        pointLabels: {
                            color: '#e2e8f0',
                            font: { size: 12, weight: '600', family: "'Plus Jakarta Sans', sans-serif" }
                        },
                        ticks: {
                            display: false,
                            min: 0,
                            max: 100,
                            stepSize: 20
                        }
                    }
                }
            }
        });
    }

    function updateRadarFingerprint() {
        if (!radarFingerprintChart) return;
        radarFingerprintChart.data.datasets[0].data = [
            fingerprintScores.visibility,
            fingerprintScores.vegetation,
            fingerprintScores.hydrological,
            fingerprintScores.urban
        ];
        radarFingerprintChart.update();
    }

    function renderSectorDonutChart(chartData) {
        const ctx = document.getElementById('sectorDonutChart').getContext('2d');
        if (sectorDonutChart) {
            sectorDonutChart.destroy();
        }

        sectorDonutChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: chartData.labels,
                datasets: [{
                    data: chartData.data,
                    backgroundColor: chartData.colors,
                    borderColor: '#0b1120',
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '72%',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleColor: '#f8fafc',
                        bodyColor: '#38bdf8',
                        borderColor: 'rgba(56, 189, 248, 0.3)',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => ` ${ctx.label}: ${ctx.raw}%`
                        }
                    }
                }
            }
        });

        // Render Custom Donut Legend
        const donutLegendEl = document.getElementById('donut-legend');
        donutLegendEl.innerHTML = '';
        chartData.labels.forEach((label, idx) => {
            const item = document.createElement('div');
            item.className = 'donut-legend-item';
            item.innerHTML = `
                <span class="donut-legend-color" style="background:${chartData.colors[idx]}"></span>
                <span>${label} (${chartData.data[idx]}%)</span>
            `;
            donutLegendEl.appendChild(item);
        });
    }

    // --- 2. Interactive Optical Comparison Split Slider & Magnifying Lens ---
    function setSliderPosition(percentage) {
        percentage = Math.max(0, Math.min(100, percentage));
        sliderDivider.style.left = `${percentage}%`;
        cloudyWrap.style.width = `${percentage}%`;
    }

    function syncOverlayImageWidth() {
        const viewportWidth = compViewport.clientWidth;
        imgCloudy.style.width = `${viewportWidth}px`;
    }

    window.addEventListener('resize', syncOverlayImageWidth);

    function handleSliderMove(e) {
        if (!isDraggingSlider && e.type !== 'click') return;
        const rect = compViewport.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const offsetX = clientX - rect.left;
        const pct = (offsetX / rect.width) * 100;
        setSliderPosition(pct);
    }

    sliderDivider.addEventListener('mousedown', (e) => {
        isDraggingSlider = true;
        e.preventDefault();
    });

    window.addEventListener('mouseup', () => { isDraggingSlider = false; });
    window.addEventListener('mousemove', handleSliderMove);

    sliderDivider.addEventListener('touchstart', (e) => {
        isDraggingSlider = true;
    }, { passive: true });
    window.addEventListener('touchend', () => { isDraggingSlider = false; });
    window.addEventListener('touchmove', handleSliderMove, { passive: true });

    compViewport.addEventListener('click', (e) => {
        if (currentViewMode !== 'split') return;
        handleSliderMove(e);
    });

    // Cloud Mask Overlay Toggle
    toggleCloudMask.addEventListener('change', (e) => {
        imgMask.style.display = e.target.checked ? 'block' : 'none';
    });

    // View Mode Switcher
    modeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            modeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentViewMode = tab.getAttribute('data-mode');

            if (currentViewMode === 'split') {
                sliderDivider.style.display = 'block';
                cloudyWrap.style.display = 'block';
                setSliderPosition(50);
            } else if (currentViewMode === 'original') {
                sliderDivider.style.display = 'none';
                cloudyWrap.style.display = 'block';
                cloudyWrap.style.width = '100%';
            } else if (currentViewMode === 'cleared') {
                sliderDivider.style.display = 'none';
                cloudyWrap.style.display = 'block';
                cloudyWrap.style.width = '0%';
            }
        });
    });

    // --- High-Definition Magnifying Lens Logic ---
    // The lens is shown on hover when the button is toggled active
    toggleMagnifierBtn.addEventListener('click', () => {
        isMagnifierEnabled = !isMagnifierEnabled;
        toggleMagnifierBtn.classList.toggle('active', isMagnifierEnabled);
        if (!isMagnifierEnabled) {
            loupeLens.style.display = 'none';
            compViewport.style.cursor = 'default';
        } else {
            compViewport.style.cursor = 'crosshair';
        }
    });

    function updateLoupePosition(e) {
        if (!isMagnifierEnabled) return;
        const rect = compViewport.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
            loupeLens.style.display = 'none';
            return;
        }

        loupeLens.style.display = 'block';

        // Smart offset: keep loupe inside viewport
        const loupeDiam = 190;
        const loupeR = loupeDiam / 2;
        let lx = x - loupeR;
        let ly = y - loupeR;
        // Clamp so loupe never overflows the viewport div
        lx = Math.max(0, Math.min(rect.width - loupeDiam, lx));
        ly = Math.max(0, Math.min(rect.height - loupeDiam, ly));
        loupeLens.style.left = `${lx}px`;
        loupeLens.style.top  = `${ly}px`;

        // Scale and position the zoomed cleared optical image inside loupe
        const zoomedW = rect.width  * currentZoomLevel;
        const zoomedH = rect.height * currentZoomLevel;
        loupeImg.style.width  = `${zoomedW}px`;
        loupeImg.style.height = `${zoomedH}px`;

        // Offset so the cursor position maps to the loupe centre
        const imgLeft = loupeR - (x * currentZoomLevel);
        const imgTop  = loupeR - (y * currentZoomLevel);
        loupeImg.style.left = `${imgLeft}px`;
        loupeImg.style.top  = `${imgTop}px`;

        loupeZoomTxt.textContent = `${currentZoomLevel.toFixed(1)}X ZOOM`;
        loupeCoordTxt.textContent = `X: ${Math.round(x)} Y: ${Math.round(y)}`;
    }

    compViewport.addEventListener('mousemove', (e) => {
        if (isDraggingSlider || !isMagnifierEnabled) return;
        updateLoupePosition(e);
    });

    compViewport.addEventListener('mouseleave', () => {
        loupeLens.style.display = 'none';
    });

    // Mouse wheel dynamically adjusts magnifying zoom level (1.5x to 5.0x)
    compViewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.deltaY < 0) {
            currentZoomLevel = Math.min(5.0, currentZoomLevel + 0.5);
        } else {
            currentZoomLevel = Math.max(1.5, currentZoomLevel - 0.5);
        }
        updateLoupePosition(e);
    }, { passive: false });

    // --- 3. Upload & File Selection Handling ---
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    changeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelected(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelected(file) {
        selectedFile = file;
        previewFilename.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (ev) => {
            previewThumb.src = ev.target.result;
            dropzoneIdle.style.display = 'none';
            dropzonePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // Enhance Button Click
    enhanceBtn.addEventListener('click', () => {
        if (!selectedFile) {
            // If no custom file, run sample scene
            loadSampleScene('agriculture');
            return;
        }
        if (isDiffCRMode) {
            uploadAndProcessFileDiffCR(selectedFile);
        } else {
            uploadAndProcessFile(selectedFile);
        }
    });

    // ── DiffCR Mode Toggle ──────────────────────────────────────────────────
    const diffcrToggle = document.getElementById('diffcr-mode-toggle');
    const diffcrBadge  = document.getElementById('diffcr-mode-badge');
    const activeModeLabel = document.getElementById('active-mode-label');
    if (diffcrToggle) {
        diffcrToggle.addEventListener('change', (e) => {
            isDiffCRMode = e.target.checked;
            if (diffcrBadge) {
                diffcrBadge.textContent = isDiffCRMode ? 'DiffCR AI ✦' : 'Fast CV';
                diffcrBadge.style.background = isDiffCRMode
                    ? 'linear-gradient(90deg,#7c3aed,#4f46e5)'
                    : 'linear-gradient(90deg,#0e7490,#0369a1)';
            }
            if (activeModeLabel) {
                activeModeLabel.innerHTML = isDiffCRMode
                    ? '<i class="fa-solid fa-brain"></i> DiffCR — T=20 Diffusion Steps (Zou et al. 2024)'
                    : '<i class="fa-solid fa-circle-info"></i> Classical OpenCV pipeline — sub-30ms';
            }
        });
    }

    async function uploadAndProcessFile(file) {
        enhanceBtn.disabled = true;
        processingIndicator.style.display = 'flex';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to process image');

            applyProcessedResult(data);
        } catch (err) {
            alert('Upload & Processing Error: ' + err.message);
        } finally {
            enhanceBtn.disabled = false;
            processingIndicator.style.display = 'none';
        }
    }

    // ── DiffCR AI Upload & Processing ──────────────────────────────────────
    async function uploadAndProcessFileDiffCR(file) {
        enhanceBtn.disabled = true;

        // Show DiffCR diffusion steps overlay
        showDiffCROverlay();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/process-diffcr', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'DiffCR processing failed');

            applyProcessedResult(data);
            // Show DiffCR badge on mode indicator
            const modeEl = document.getElementById('active-mode-label');
            if (modeEl) modeEl.textContent = 'DiffCR AI (T=20 Steps)';
        } catch (err) {
            alert('DiffCR Processing Error: ' + err.message);
        } finally {
            enhanceBtn.disabled = false;
            hideDiffCROverlay();
        }
    }

    // ── DiffCR Animated Steps Overlay ─────────────────────────────────────
    let _diffcrOverlayTimer = null;
    function showDiffCROverlay() {
        let overlay = document.getElementById('diffcr-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'diffcr-overlay';
            overlay.style.cssText = [
                'position:fixed;inset:0;z-index:9999;',
                'display:flex;flex-direction:column;align-items:center;justify-content:center;',
                'background:rgba(5,8,22,0.92);backdrop-filter:blur(12px);',
                'font-family:\'Plus Jakarta Sans\',sans-serif;color:#f1f5f9;'
            ].join('');
            overlay.innerHTML = `
                <div style="text-align:center;max-width:420px;padding:2rem;">
                    <div style="font-size:2.8rem;margin-bottom:0.6rem;">🤖</div>
                    <div style="font-size:1.4rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.4rem;">DiffCR Engine Active</div>
                    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:1.4rem;">Conditional Diffusion Framework for Cloud Removal<br><span style="color:#64748b;">(Zou et al., IEEE TGRS 2024)</span></div>
                    <div id="diffcr-step-label" style="font-size:0.9rem;color:#38bdf8;margin-bottom:0.7rem;font-weight:600;">Initializing...</div>
                    <div style="width:320px;height:6px;background:#1e293b;border-radius:99px;overflow:hidden;margin:0 auto 1rem;">
                        <div id="diffcr-progress-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#7c3aed,#06b6d4);border-radius:99px;transition:width 0.18s ease;"></div>
                    </div>
                    <div id="diffcr-step-counter" style="font-size:0.75rem;color:#64748b;">Step 0 / 20</div>
                    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;justify-content:center;margin-top:1.2rem;" id="diffcr-steps-grid"></div>
                </div>
            `;
            document.body.appendChild(overlay);

            // Build step dots
            const grid = overlay.querySelector('#diffcr-steps-grid');
            for (let i = 0; i < 20; i++) {
                const dot = document.createElement('div');
                dot.id = `diffcr-dot-${i}`;
                dot.style.cssText = 'width:10px;height:10px;border-radius:50%;background:#1e293b;transition:background 0.15s ease,transform 0.15s ease;';
                grid.appendChild(dot);
            }
        }
        overlay.style.display = 'flex';

        // Animate steps
        const steps = [
            'Estimating Cloud Mask...', 'Forward Diffusion (β schedule)...',
            'Denoising t=19...', 'Denoising t=17...', 'Denoising t=15...',
            'Denoising t=13...', 'Denoising t=11...', 'Conditional Synthesis t=9...',
            'Refining t=7...', 'Spectral Propagation t=5...', 'Laplacian Sharpening...',
            'Reflectance Recovery...', 'CLAHE Enhancement...', 'Gamma Correction...',
            'Multi-Scale Pyramid Blend...', 'Water Body Refinement...', 'Vegetation Bands...',
            'Final Denoising Pass...', 'Histogram Re-balance...', 'Output Render ✓'
        ];
        let currentStep = 0;
        const progressBar = overlay.querySelector('#diffcr-progress-bar');
        const stepLabel   = overlay.querySelector('#diffcr-step-label');
        const stepCounter = overlay.querySelector('#diffcr-step-counter');

        _diffcrOverlayTimer = setInterval(() => {
            if (currentStep >= 20) return;
            // Light up dot
            const dot = overlay.querySelector(`#diffcr-dot-${currentStep}`);
            if (dot) { dot.style.background = '#7c3aed'; dot.style.transform = 'scale(1.3)'; }
            stepLabel.textContent   = steps[currentStep] || 'Processing...';
            stepCounter.textContent = `Step ${currentStep + 1} / 20`;
            progressBar.style.width  = `${((currentStep + 1) / 20) * 100}%`;
            currentStep++;
        }, 160);  // ~3.2s total animation
    }

    function hideDiffCROverlay() {
        clearInterval(_diffcrOverlayTimer);
        const overlay = document.getElementById('diffcr-overlay');
        if (overlay) overlay.style.display = 'none';
    }

    // 1-Click Sample Scene Buttons
    const sampleButtons = document.querySelectorAll('.sample-btn');
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const sceneId = btn.getAttribute('data-scene');
            loadSampleScene(sceneId);
        });
    });

    async function loadSampleScene(sceneId) {
        enhanceBtn.disabled = true;
        processingIndicator.style.display = 'flex';

        try {
            const res = await fetch(`/sample/${sceneId}`, { method: 'POST' });
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to load sample scene');

            // Update preview card thumbnail
            previewThumb.src = data.input_image_url;
            previewFilename.textContent = `${sceneId.toUpperCase()} Satellite Scene`;
            dropzoneIdle.style.display = 'none';
            dropzonePreview.style.display = 'block';

            applyProcessedResult(data);
        } catch (err) {
            alert('Error loading sample scene: ' + err.message);
        } finally {
            enhanceBtn.disabled = false;
            processingIndicator.style.display = 'none';
        }
    }

    // --- 4. Apply Processed Result & Update Telemetry ---
    function applyProcessedResult(data) {
        currentSceneData = data;
        currentClearedFilename = data.filename;

        // 1. Update Images in Optical Comparison Viewport
        imgCloudy.src = data.input_image_url;
        imgCleared.src = data.output_image_url;
        loupeImg.src = data.output_image_url; // Crystal-clear enhanced optical view inside magnifying lens
        if (data.mask_image_url) {
            imgMask.src = data.mask_image_url;
        }

        // Wait for image load to lock dimensions
        imgCloudy.onload = () => {
            syncOverlayImageWidth();
            setSliderPosition(50);
        };
        imgCleared.onload = () => {
            syncOverlayImageWidth();
            setSliderPosition(50);
        };

        // 2. Download button link
        downloadClearedBtn.href = `/download/${data.filename}`;

        // 3. Update KPI Badges
        const m = data.metrics;
        valProcTime.textContent = `${m.processing_time}s`;
        valInputSize.textContent = `${m.input_size_kb} KB`;
        valOutputSize.textContent = `${m.output_size_kb} KB`;
        valContrastBoost.textContent = `+${m.contrast_boost_pct}%`;
        valHazeReduction.textContent = `${m.haze_reduction_pct}%`;

        // 4. Update Cloud Coverage Bar & Note
        valCloudCoverage.textContent = `${m.cloud_coverage_pct}%`;
        coverageBarFill.style.width = `${Math.min(100, m.cloud_coverage_pct)}%`;
        if (m.cloud_coverage_pct < 10) {
            coverageNote.textContent = 'Clear Optical Window (Optimal)';
            coverageNote.style.color = 'var(--emerald-accent)';
        } else if (m.cloud_coverage_pct < 40) {
            coverageNote.textContent = 'Moderate Atmospheric Haze Detected';
            coverageNote.style.color = 'var(--amber-accent)';
        } else {
            coverageNote.textContent = 'Heavy Cloud Occlusion Penetrated';
            coverageNote.style.color = 'var(--cyan-accent)';
        }

        // 5. Update Histogram Chart
        if (m.histogram && histogramChart) {
            histogramChart.data.labels = m.histogram.labels;
            histogramChart.data.datasets[0].data = m.histogram.before;
            histogramChart.data.datasets[1].data = m.histogram.after;
            histogramChart.update();
        }

        // 6. Update Radar Fingerprint Visibility
        fingerprintScores.visibility = Math.min(100, Math.round(m.haze_reduction_pct * 1.1));
        updateRadarFingerprint();

        // 7. Auto-run active sector module
        runSectorAnalysis(activeSectorModule);
    }

    // --- 5. Multi-Sector Post-Processing CV Suite ---
    sectorTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sectorTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeSectorModule = btn.getAttribute('data-module');
            runSectorAnalysis(activeSectorModule);
        });
    });

    async function runSectorAnalysis(moduleName) {
        if (!currentClearedFilename) return;

        sectorSpinner.style.display = 'flex';

        try {
            const res = await fetch(`/analyze/${moduleName}`, { method: 'POST' });
            const result = await res.json();
            if (!result.success) throw new Error(result.error || 'Sector analysis failed');

            const d = result.data;

            // 1. Update Analysis Map Image
            analysisResultMap.src = result.analysis_image_url;

            // 2. Update Header & Description
            moduleDisplayTitle.textContent = d.title;
            const descriptions = {
                ndvi: 'NDVI quantifies vegetation health by measuring simulated Near-Infrared reflectance against red band absorption. Green indicates thriving crops/forests.',
                flood: 'Identifies open surface water bodies, waterlogging extent, and hydrological inundation perimeters for rapid disaster response.',
                flood_rescue: '🚨 Cross-module rescue intelligence: detects flood water zones AND buildings within those zones. Red SOS markers = structures submerged and requiring immediate rescue. Amber = at-risk perimeter buildings. Green = safe structures.',
                buildings: 'Extracts structural edges, building footprints, and cadastral density metrics for municipal planning and urban development.',
                landcover: 'Classifies the scene into 4 distinct bio-climatic terrain regimes (Vegetation, Water, Bare Soil, Urban) for carbon sink monitoring.'
            };
            moduleDisplayDesc.textContent = descriptions[moduleName] || d.domain;

            // 3. Render Stats List
            sectorStatsList.innerHTML = '';
            for (const [key, val] of Object.entries(d)) {
                if (['module_id', 'title', 'domain', 'chart'].includes(key)) continue;

                const row = document.createElement('div');
                row.className = 'stat-item-row';

                // Color-code rescue stat rows
                if (key === 'rescue_critical_count') row.classList.add('rescue-critical');
                if (key === 'at_risk_count') row.classList.add('at-risk');

                const labelText = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                // Smart unit detection: add % only if value is a plain number and key has 'pct' or 'percent'
                let displayVal;
                if (typeof val === 'number') {
                    const isPercent = key.includes('pct') || key.includes('percent');
                    displayVal = isPercent ? `${val}%` : val;
                } else {
                    displayVal = val; // Already has units (e.g., '82.1 MT/ha', '4.5 km²')
                }

                row.innerHTML = `
                    <span class="stat-item-label">${labelText}</span>
                    <span class="stat-item-val">${displayVal}</span>
                `;
                sectorStatsList.appendChild(row);
            }

            // 4. Render Donut Chart
            if (d.chart) {
                renderSectorDonutChart(d.chart);
            }

            // 5. Update Radar Fingerprint Based on Results
            if (moduleName === 'ndvi') {
                fingerprintScores.vegetation = Math.min(100, Math.round((d.dense_healthy_pct || 30) * 1.5));
            } else if (moduleName === 'flood') {
                fingerprintScores.hydrological = Math.min(100, Math.round((d.water_coverage_percent || 15) * 2.5));
            } else if (moduleName === 'flood_rescue') {
                fingerprintScores.hydrological = Math.min(100, Math.round((d.flood_coverage_percent || 15) * 2.5));
                fingerprintScores.urban = Math.min(100, Math.round(((d.rescue_critical_count || 0) + (d.at_risk_count || 0)) * 8));
                // Flash rescue alert badge if critical structures found
                if ((d.rescue_critical_count || 0) > 0) {
                    const statsDiv = document.getElementById('sector-stats-list');
                    if (statsDiv) {
                        const alertBanner = document.createElement('div');
                        alertBanner.style.cssText = [
                            'background:linear-gradient(90deg,rgba(239,68,68,0.18),rgba(239,68,68,0.05));',
                            'border:1px solid rgba(239,68,68,0.5);border-radius:8px;padding:0.55rem 0.75rem;',
                            'margin-bottom:0.6rem;display:flex;align-items:center;gap:0.5rem;',
                            'font-size:0.8rem;font-weight:700;color:#ef4444;animation:pulse-red 1.5s infinite;'
                        ].join('');
                        alertBanner.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> RESCUE ALERT: ${d.rescue_critical_count} STRUCTURE(S) SUBMERGED — Immediate evacuation required!`;
                        statsDiv.prepend(alertBanner);
                    }
                }
            } else if (moduleName === 'buildings') {
                fingerprintScores.urban = Math.min(100, Math.round((d.builtup_density_percent || 20) * 2.0));
            } else if (moduleName === 'landcover') {
                fingerprintScores.vegetation = Math.min(100, Math.round((d.vegetation_forest_percent || 35) * 1.3));
                fingerprintScores.hydrological = Math.min(100, Math.round((d.water_bodies_percent || 10) * 2.0));
                fingerprintScores.urban = Math.min(100, Math.round((d.urban_builtup_percent || 15) * 1.8));
            }
            updateRadarFingerprint();

        } catch (err) {
            console.error('Sector analysis error:', err);
        } finally {
            sectorSpinner.style.display = 'none';
        }
    }

    // Export Analysis Map & Report Action
    exportAnalysisBtn.addEventListener('click', () => {
        if (analysisResultMap.src) {
            const a = document.createElement('a');
            a.href = analysisResultMap.src;
            a.download = `analysis_${activeSectorModule}_map.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    });

    // --- 6. Export PDF Report ---
    exportPdfBtn.addEventListener('click', () => {
        window.location.href = '/export-pdf';
    });

    // --- 7. Mission Telemetry Dashboard Modal ---
    openDashboardBtn.addEventListener('click', async () => {
        dashboardModal.style.display = 'flex';
        try {
            const res = await fetch('/dashboard-stats');
            const data = await res.json();
            if (data.success) {
                const s = data.stats;
                document.getElementById('dash-total-processed').textContent = s.total_processed;
                document.getElementById('dash-avg-contrast').textContent = `+${s.avg_contrast_boost}%`;
                document.getElementById('dash-total-land').textContent = `${s.total_land_analyzed_sqkm} km²`;
                document.getElementById('dash-flood-alerts').textContent = `${s.flood_alerts_detected} Alerts`;
            }
        } catch (e) {
            console.warn('Could not refresh dashboard stats:', e);
        }
    });

    closeDashboardBtn.addEventListener('click', () => {
        dashboardModal.style.display = 'none';
    });

    dashboardModal.addEventListener('click', (e) => {
        if (e.target === dashboardModal) {
            dashboardModal.style.display = 'none';
        }
    });

    // --- 8. Initial Startup Execution ---
    initHistogramChart();
    initRadarFingerprintChart();

    // Auto-load default Agricultural Scene on initial start
    loadSampleScene('agriculture');
});
