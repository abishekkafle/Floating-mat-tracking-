const $ = (id) => document.getElementById(id);

function getBackendUrl() {
  return $('backend-url').value.replace(/\/$/, '');
}

function showError(msg) {
  const el = $('error-banner');
  el.textContent = msg;
  el.classList.remove('hidden');
}
function clearError() {
  $('error-banner').classList.add('hidden');
}

async function apiGet(path) {
  try {
    const r = await fetch(getBackendUrl() + path);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

async function apiPost(path, body) {
  try {
    const r = await fetch(getBackendUrl() + path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (e) { /* not json */ }
      showError(`Backend: ${detail}`);
      return null;
    }
    return await r.json();
  } catch (e) {
    showError(`Could not reach backend at ${getBackendUrl()}: ${e}`);
    return null;
  }
}

function setButtonsDisabled(disabled) {
  ['start-btn', 'stop-btn', 'set-ref-btn', 'clear-btn'].forEach((id) => { $(id).disabled = disabled; });
}

// --------------------------------------------------------------- settings

const PRESETS = {
  default: { h_min: 90, s_min: 80, v_min: 80, h_max: 140, s_max: 255, v_max: 255 },
  light: { h_min: 100, s_min: 50, v_min: 50, h_max: 140, s_max: 255, v_max: 255 },
  dark: { h_min: 90, s_min: 100, v_min: 50, h_max: 140, s_max: 255, v_max: 200 },
};
const DETECTION_SLIDER_KEYS = ['h_min', 's_min', 'v_min', 'h_max', 's_max', 'v_max', 'area_min', 'poly_eps', 'morph_k'];

let settingsDebounce = null;
function scheduleSettingsPush() {
  clearTimeout(settingsDebounce);
  settingsDebounce = setTimeout(pushSettings, 150);
}

async function pushSettings() {
  await apiPost('/api/settings/detection', {
    h_min: +$('h_min').value, s_min: +$('s_min').value, v_min: +$('v_min').value,
    h_max: +$('h_max').value, s_max: +$('s_max').value, v_max: +$('v_max').value,
    area_min: +$('area_min').value, poly_eps_pct: +$('poly_eps').value, morph_k: +$('morph_k').value,
    debug: $('debug_mode').checked,
  });
  await apiPost('/api/settings/camera', {
    index: +$('cam_index').value, width: +$('frame_width').value, height: +$('frame_height').value,
    processing_fps: +$('processing_fps').value, known_side_cm: +$('known_side').value,
  });
}

function initSettingsControls() {
  DETECTION_SLIDER_KEYS.forEach((key) => {
    const el = $(key);
    el.addEventListener('input', () => {
      const label = $('v-' + key);
      if (label) label.textContent = el.value;
      scheduleSettingsPush();
    });
  });
  $('processing_fps').addEventListener('input', () => {
    $('v-processing_fps').textContent = $('processing_fps').value;
    scheduleSettingsPush();
  });
  $('debug_mode').addEventListener('change', scheduleSettingsPush);
  ['cam_index', 'frame_width', 'frame_height', 'known_side'].forEach((key) => {
    $(key).addEventListener('change', scheduleSettingsPush);
  });
  $('preset-select').addEventListener('change', () => {
    const preset = PRESETS[$('preset-select').value];
    for (const k in preset) {
      $(k).value = preset[k];
      const label = $('v-' + k);
      if (label) label.textContent = preset[k];
    }
    scheduleSettingsPush();
  });
}

// ----------------------------------------------------------------- buttons

function initButtons() {
  $('start-btn').addEventListener('click', async () => {
    await pushSettings();
    await apiPost('/api/start');
  });
  $('stop-btn').addEventListener('click', () => apiPost('/api/stop'));
  $('set-ref-btn').addEventListener('click', () => apiPost('/api/reference/set'));
  $('clear-btn').addEventListener('click', () => apiPost('/api/reference/clear'));

  $('test-camera-btn').addEventListener('click', async () => {
    const params = new URLSearchParams({
      index: +$('cam_index').value, width: +$('frame_width').value, height: +$('frame_height').value,
    });
    const resultEl = $('test-camera-result');
    resultEl.innerHTML = 'Testing…';
    try {
      const r = await fetch(`${getBackendUrl()}/api/camera/test?${params}`);
      if (r.ok) {
        const blob = await r.blob();
        resultEl.innerHTML = '';
        const img = document.createElement('img');
        img.src = URL.createObjectURL(blob);
        resultEl.appendChild(img);
      } else {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* not json */ }
        resultEl.innerHTML = `<div class="msg">${detail}</div>`;
      }
    } catch (e) {
      resultEl.innerHTML = `<div class="msg">Could not reach backend: ${e}</div>`;
    }
  });

  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      $('tab-distance').classList.toggle('hidden', tab !== 'distance');
      $('tab-angle').classList.toggle('hidden', tab !== 'angle');
    });
  });

  $('download-csv-btn').addEventListener('click', () => {
    if (!latestHistory.length) return;
    const t0 = latestHistory[0].t;
    const header = 'time_s,dx_cm,dy_cm,distance_cm,angle_drift_deg\n';
    const rows = latestHistory.map((h) => [
      (h.t - t0).toFixed(3), h.dx_cm ?? '', h.dy_cm ?? '', h.distance_cm ?? '', h.angle_drift_deg ?? '',
    ].join(',')).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'hex_mat_tracking_cm.csv';
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ------------------------------------------------------------ status poll

let lastVideoRunning = false;

async function pollStatus() {
  const status = await apiGet('/api/status');
  const badge = $('backend-badge');
  if (!status) {
    badge.textContent = 'unreachable';
    badge.className = 'badge badge-bad';
    setButtonsDisabled(true);
    return;
  }
  badge.textContent = 'connected';
  badge.className = 'badge badge-ok';
  setButtonsDisabled(false);

  if (status.error) showError(`Backend reported: ${status.error}`);
  else clearError();

  $('status-running').innerHTML = (status.running ? '\u{1F7E2}' : '\u{1F534}') + ` Status: ${status.running ? 'Running' : 'Stopped'}`;
  $('status-ref').innerHTML = `\u{1F4CD} Reference: ${status.ref_set ? 'Set' : 'Not Set'}`;
  $('status-mat').innerHTML = `\u{1F537} Mat: ${status.detected ? 'Detected' : (status.running ? 'Not Detected' : 'Not Running')}`;

  $('metric-distance').textContent = status.distance_cm != null ? `${status.distance_cm.toFixed(1)} cm` : '—';
  $('metric-angle').textContent = status.angle_drift_deg != null ? `${status.angle_drift_deg.toFixed(1)} deg` : '—';

  const img = $('video-feed');
  const placeholder = $('video-placeholder');
  if (status.running && !lastVideoRunning) {
    img.src = `${getBackendUrl()}/api/video_feed?_=${Date.now()}`;
    img.classList.add('showing');
    placeholder.classList.add('hidden');
  } else if (!status.running && lastVideoRunning) {
    img.removeAttribute('src');
    img.classList.remove('showing');
    placeholder.classList.remove('hidden');
  }
  lastVideoRunning = status.running;
}

// ----------------------------------------------------------------- charts

const tooltip = $('tooltip');
function showTip(evt, html) {
  tooltip.innerHTML = html;
  tooltip.style.left = evt.clientX + 'px';
  tooltip.style.top = evt.clientY + 'px';
  tooltip.classList.add('show');
}
function hideTip() { tooltip.classList.remove('show'); }

function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function renderTimeSeries(svg, points, color, unit, decimals) {
  const W = 640, H = 240, padL = 46, padR = 20, padT = 16, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  if (!points.length) {
    const t = svgEl('text', { class: 'axis-label', x: W / 2, y: H / 2, 'text-anchor': 'middle' });
    t.textContent = 'No data yet — start the stream and set a reference.';
    svg.appendChild(t);
    return;
  }

  const xs = points.map((p) => p.t);
  const ys = points.map((p) => p.v);
  const xMin = Math.min(...xs), xMax = Math.max(...xs) || 1;
  const yMaxRaw = Math.max(...ys, 0);
  const yMinRaw = Math.min(...ys, 0);
  const yMax = yMaxRaw * 1.15 || 1;
  const yMin = yMinRaw < 0 ? yMinRaw * 1.15 : 0;
  const xPos = (x) => padL + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotW);
  const yPos = (v) => padT + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

  for (let i = 0; i <= 4; i++) {
    const v = yMin + ((yMax - yMin) * i) / 4;
    const gy = yPos(v);
    svg.appendChild(svgEl('line', { class: 'gridline', x1: padL, x2: W - padR, y1: gy, y2: gy }));
    const t = svgEl('text', { class: 'axis-label', x: padL - 6, y: gy + 3, 'text-anchor': 'end' });
    t.textContent = v.toFixed(decimals);
    svg.appendChild(t);
  }
  [xMin, xMax].forEach((x, i) => {
    const t = svgEl('text', { class: 'axis-label', x: xPos(x), y: H - 8, 'text-anchor': i === 0 ? 'start' : 'end' });
    t.textContent = `${x.toFixed(0)}s`;
    svg.appendChild(t);
  });

  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xPos(p.t)} ${yPos(p.v)}`).join(' ');
  svg.appendChild(svgEl('path', { class: 'line-path', d, stroke: color }));

  const hoverLine = svgEl('line', { x1: -100, x2: -100, y1: padT, y2: padT + plotH, stroke: color, 'stroke-opacity': 0.35 });
  const hoverDot = svgEl('circle', { class: 'dot', r: 4, fill: color, cx: -100, cy: -100 });
  svg.appendChild(hoverLine);
  svg.appendChild(hoverDot);

  const hit = svgEl('rect', { x: padL, y: padT, width: plotW, height: plotH, fill: 'transparent' });
  hit.addEventListener('mousemove', (evt) => {
    const rect = svg.getBoundingClientRect();
    const mouseXsvg = ((evt.clientX - rect.left) / rect.width) * W;
    const frac = Math.min(1, Math.max(0, (mouseXsvg - padL) / plotW));
    const targetX = xMin + frac * (xMax - xMin);
    let idx = 0, best = Infinity;
    points.forEach((p, i) => { const diff = Math.abs(p.t - targetX); if (diff < best) { best = diff; idx = i; } });
    const p = points[idx];
    const cx = xPos(p.t), cy = yPos(p.v);
    hoverDot.setAttribute('cx', cx); hoverDot.setAttribute('cy', cy);
    hoverLine.setAttribute('x1', cx); hoverLine.setAttribute('x2', cx);
    showTip(evt, `<b>${p.t.toFixed(0)}s</b><br>${p.v.toFixed(decimals)} ${unit}`);
  });
  hit.addEventListener('mouseleave', () => {
    hoverLine.setAttribute('x1', -100); hoverLine.setAttribute('x2', -100);
    hoverDot.setAttribute('cx', -100);
    hideTip();
  });
  svg.appendChild(hit);
}

let latestHistory = [];
async function pollHistory() {
  const hist = await apiGet('/api/history');
  if (!hist) return;
  latestHistory = hist;
  const t0 = hist.length ? hist[0].t : 0;
  const distPoints = hist.filter((h) => h.distance_cm != null).map((h) => ({ t: h.t - t0, v: h.distance_cm }));
  const anglePoints = hist.filter((h) => h.angle_drift_deg != null).map((h) => ({ t: h.t - t0, v: h.angle_drift_deg }));
  renderTimeSeries($('chart-distance'), distPoints, '#0072B2', 'cm', 1);
  renderTimeSeries($('chart-angle'), anglePoints, '#D55E00', 'deg', 1);
}

// ------------------------------------------------------------------- init

initSettingsControls();
initButtons();
pollStatus();
pollHistory();
setInterval(pollStatus, 1000);
setInterval(pollHistory, 2000);
