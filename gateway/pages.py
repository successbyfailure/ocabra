"""HTML page generators for the gateway: loading, directory, disabled, and not-found."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared CSS / design tokens (dark theme matching oCabra UI palette)
# ---------------------------------------------------------------------------

_BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0a; color: #e5e7eb; min-height: 100vh;
}
a { color: #60a5fa; text-decoration: none; }
a:hover { text-decoration: underline; }
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 9999px; font-size: 12px; font-weight: 500;
}
.badge-active   { background: #14532d; color: #4ade80; }
.badge-idle     { background: #1c1917; color: #a8a29e; }
.badge-disabled { background: #1c1917; color: #6b7280; }
.badge-unreach  { background: #450a0a; color: #f87171; }
.badge-starting { background: #1e3a5f; color: #60a5fa; }
.dot {
    width: 7px; height: 7px; border-radius: 50%; display: inline-block;
}
.dot-active   { background: #4ade80; }
.dot-idle     { background: #6b7280; }
.dot-disabled { background: #4b5563; }
.dot-unreach  { background: #f87171; }
.dot-starting { background: #60a5fa; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 6px; font-size: 13px;
    font-weight: 500; cursor: pointer; border: none; transition: opacity .15s;
}
button:disabled { opacity: .4; cursor: not-allowed; }
"""

# ---------------------------------------------------------------------------
# Loading page
# ---------------------------------------------------------------------------

def loading_page(service_id: str, display_name: str, startup_timeout_s: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Starting {display_name}…</title>
<style>
{_BASE_CSS}
body {{ display: flex; flex-direction: column; align-items: center;
       justify-content: center; padding: 24px; }}
.card {{
    width: 100%; max-width: 480px;
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 36px 32px; text-align: center;
}}
.icon {{ font-size: 40px; margin-bottom: 16px; }}
h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 8px; color: #f9fafb; }}
.subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 32px; }}
.spinner {{
    width: 44px; height: 44px; margin: 0 auto 24px;
    border: 3px solid #1f2937; border-top-color: #3b82f6;
    border-radius: 50%; animation: spin 0.9s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.status {{ font-size: 14px; color: #9ca3af; margin-bottom: 8px; min-height: 20px; }}
.detail {{ font-size: 12px; color: #6b7280; min-height: 16px; }}
.elapsed {{ font-size: 11px; color: #4b5563; margin-top: 24px; }}
.error {{ color: #f87171; font-size: 13px; margin-top: 16px; display: none; }}
.ready {{ color: #4ade80; font-size: 14px; margin-top: 16px; display: none; }}
</style>
</head>
<body>
<div class="card">
  <div class="icon">🚀</div>
  <h1>{display_name}</h1>
  <p class="subtitle">Service is starting up</p>
  <div class="spinner" id="spinner"></div>
  <p class="status" id="status-text">Initialising…</p>
  <p class="detail" id="detail-text"></p>
  <p class="elapsed" id="elapsed"></p>
  <p class="error" id="error-text"></p>
  <p class="ready" id="ready-text">✓ Ready — connecting…</p>
</div>
<script>
const SERVICE_ID = {service_id!r};
const TIMEOUT_S  = {startup_timeout_s};
const start      = Date.now();
let attempts     = 0;

function setStatus(text, detail) {{
  document.getElementById('status-text').textContent = text;
  document.getElementById('detail-text').textContent = detail || '';
}}

function tick() {{
  const elapsed = Math.round((Date.now() - start) / 1000);
  document.getElementById('elapsed').textContent = `${{elapsed}}s elapsed`;
}}

async function poll() {{
  tick();
  const elapsed = (Date.now() - start) / 1000;

  if (elapsed > TIMEOUT_S) {{
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('error-text').style.display = 'block';
    document.getElementById('error-text').textContent =
      'Timeout — service did not start within ' + TIMEOUT_S + 's. Check logs.';
    return;
  }}

  try {{
    const r = await fetch('/_gw/status/' + SERVICE_ID);
    if (!r.ok) {{ throw new Error('HTTP ' + r.status); }}
    const d = await r.json();

    if (d.service_alive) {{
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('ready-text').style.display = 'block';
      setStatus('Service is ready', '');
      setTimeout(() => location.reload(), 600);
      return;
    }}

    const statusMap = {{
      unreachable: 'Waiting for service to respond…',
      idle:        'Service loaded, waiting…',
      active:      'Service active',
      restarting:  'Container restarting…',
      disabled:    'Service is disabled',
    }};
    setStatus(statusMap[d.status] || d.status, d.detail || '');

    if (d.status === 'disabled') {{
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('error-text').style.display = 'block';
      document.getElementById('error-text').textContent =
        'Service is disabled. Enable it from the oCabra dashboard.';
      return;
    }}
  }} catch (e) {{
    setStatus('Checking…', e.message);
  }}

  setTimeout(poll, 2000);
}}

poll();
setInterval(tick, 1000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Disabled page
# ---------------------------------------------------------------------------

def disabled_page(display_name: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{display_name} — Disabled</title>
<style>
{_BASE_CSS}
body {{ display: flex; align-items: center; justify-content: center; padding: 24px; }}
.card {{
    max-width: 420px; text-align: center;
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 36px 32px;
}}
h1 {{ font-size: 20px; font-weight: 600; color: #f9fafb; margin: 12px 0 8px; }}
p  {{ color: #6b7280; font-size: 14px; line-height: 1.6; }}
</style>
</head>
<body>
<div class="card">
  <div style="font-size:40px">🔒</div>
  <h1>{display_name}</h1>
  <p>This service is currently disabled.<br>
     Enable it from the <a href="javascript:history.back()">oCabra dashboard</a>.</p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Not-found page
# ---------------------------------------------------------------------------

def not_found_page(host: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Not found</title>
<style>{_BASE_CSS}
body {{ display:flex; align-items:center; justify-content:center; padding:24px; }}
.card {{ max-width:380px; text-align:center; background:#111827;
         border:1px solid #1f2937; border-radius:12px; padding:36px 32px; }}
h1 {{ font-size:20px; font-weight:600; color:#f9fafb; margin:12px 0 8px; }}
p  {{ color:#6b7280; font-size:14px; }}
</style>
</head>
<body>
<div class="card">
  <div style="font-size:40px">🔍</div>
  <h1>No service found</h1>
  <p>No generation service is mapped to <code style="color:#e5e7eb">{host}</code>.</p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Directory page
# ---------------------------------------------------------------------------

_STATUS_BADGE = {
    "active":      ('<span class="dot dot-active"></span>',   "badge-active",   "Active"),
    "idle":        ('<span class="dot dot-idle"></span>',     "badge-idle",     "Idle"),
    "disabled":    ('<span class="dot dot-disabled"></span>', "badge-disabled", "Disabled"),
    "unreachable": ('<span class="dot dot-unreach"></span>',  "badge-unreach",  "Offline"),
    "restarting":  ('<span class="dot dot-starting"></span>', "badge-starting", "Restarting"),
}

_GPU_LABEL = {0: "RTX 3060 12G", 1: "RTX 3090 24G"}


def directory_page() -> str:
    """Directory page with login gate and live service status."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>oCabra — Generation Services</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0a; color: #e5e7eb; min-height: 100vh; padding: 32px 16px;
}
/* ── Login overlay ── */
#login-overlay {
    display: none; position: fixed; inset: 0; z-index: 100;
    background: #0a0a0a; align-items: center; justify-content: center;
}
#login-overlay.active { display: flex; }
.login-card {
    width: 100%; max-width: 360px;
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 36px 32px;
}
.login-card h1 { font-size: 20px; font-weight: 700; color: #f9fafb; margin-bottom: 4px; }
.login-card p  { font-size: 13px; color: #6b7280; margin-bottom: 24px; }
.field { margin-bottom: 16px; }
.field label { display: block; font-size: 13px; color: #9ca3af; margin-bottom: 6px; }
.field input {
    width: 100%; padding: 9px 12px; background: #1f2937;
    border: 1px solid #374151; border-radius: 6px; color: #f9fafb;
    font-size: 14px; outline: none;
}
.field input:focus { border-color: #3b82f6; }
.login-btn {
    width: 100%; padding: 10px; background: #2563eb; color: #fff;
    border: none; border-radius: 6px; font-size: 14px; font-weight: 500;
    cursor: pointer; margin-top: 4px;
}
.login-btn:hover { background: #1d4ed8; }
.login-btn:disabled { opacity: .5; cursor: not-allowed; }
.login-error { color: #f87171; font-size: 13px; margin-top: 10px; min-height: 18px; }
/* ── Main ── */
#main { display: none; }
#main.active { display: block; }
header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 40px; max-width: 960px; margin-left: auto; margin-right: auto;
    padding-bottom: 24px; border-bottom: 1px solid #1f2937;
}
.header-left h1 { font-size: 22px; font-weight: 700; color: #f9fafb; }
.header-left p  { color: #6b7280; font-size: 13px; margin-top: 4px; }
.user-bar { display: flex; align-items: center; gap: 12px; }
.user-chip {
    font-size: 13px; color: #9ca3af; background: #1f2937;
    padding: 4px 12px; border-radius: 20px;
}
.logout-btn {
    font-size: 12px; color: #6b7280; background: none; border: none;
    cursor: pointer; padding: 4px 8px; border-radius: 4px;
}
.logout-btn:hover { color: #f9fafb; background: #1f2937; }
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px; max-width: 960px; margin: 0 auto;
}
.card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 24px;
    display: flex; flex-direction: column; gap: 12px;
    transition: border-color .15s;
}
.card:hover { border-color: #374151; }
.card-header { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.card-name { font-size: 16px; font-weight: 600; color: #f9fafb; }
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 10px; border-radius: 9999px; font-size: 12px; font-weight: 500;
}
.badge-active   { background: #14532d; color: #4ade80; }
.badge-idle     { background: #1c1917; color: #a8a29e; }
.badge-disabled { background: #1c1917; color: #6b7280; }
.badge-unreach  { background: #450a0a; color: #f87171; }
.badge-starting { background: #1e3a5f; color: #60a5fa; }
.dot { width: 7px; height: 7px; border-radius: 50%; }
.dot-active   { background: #4ade80; }
.dot-idle     { background: #6b7280; }
.dot-disabled { background: #4b5563; }
.dot-unreach  { background: #f87171; }
.dot-starting { background: #60a5fa; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.meta { display: flex; gap: 8px; flex-wrap: wrap; }
.tag {
    font-size: 11px; color: #6b7280; background: #1f2937;
    padding: 2px 8px; border-radius: 4px;
}
.card-desc { font-size: 12px; color: #9ca3af; line-height: 1.5; }
.detail { font-size: 12px; color: #4b5563; min-height: 16px; }
.actions { display: flex; gap: 8px; margin-top: 4px; }
.btn {
    flex: 1; padding: 7px 12px; border-radius: 6px; font-size: 13px;
    font-weight: 500; cursor: pointer; border: none; transition: opacity .15s; text-align: center;
}
.btn:disabled { opacity: .35; cursor: not-allowed; }
.btn-primary { background: #2563eb; color: #fff; }
.btn-primary:hover:not(:disabled) { background: #1d4ed8; }
.btn-danger  { background: #991b1b; color: #fca5a5; }
.btn-danger:hover:not(:disabled)  { background: #7f1d1d; }
.btn-open    { background: #064e3b; color: #6ee7b7; text-decoration: none;
               display: inline-flex; align-items: center; justify-content: center; }
.btn-open:hover { background: #065f46; }
.footer { text-align: center; margin-top: 48px; font-size: 12px; color: #374151; }
.loading-msg { text-align:center; color:#4b5563; margin-top:60px; font-size:14px; }
</style>
</head>
<body>

<!-- Login overlay -->
<div id="login-overlay">
  <div class="login-card">
    <h1>oCabra</h1>
    <p>Inicia sesión para acceder a Generation Services</p>
    <div class="field">
      <label>Usuario</label>
      <input type="text" id="login-user" autocomplete="username" placeholder="usuario" />
    </div>
    <div class="field">
      <label>Contraseña</label>
      <input type="password" id="login-pass" autocomplete="current-password" placeholder="••••••••" />
    </div>
    <button class="login-btn" id="login-btn" onclick="doLogin()">Entrar</button>
    <p class="login-error" id="login-error"></p>
  </div>
</div>

<!-- Main content -->
<div id="main">
  <header>
    <div class="header-left">
      <h1>⚡ Generation Services</h1>
      <p id="refresh-hint">Cargando…</p>
    </div>
    <div class="user-bar">
      <span class="user-chip" id="user-chip"></span>
      <button class="logout-btn" onclick="doLogout()">Cerrar sesión</button>
    </div>
  </header>
  <div class="grid" id="grid">
    <p class="loading-msg">Obteniendo estado de servicios…</p>
  </div>
  <div class="footer">oCabra gateway — actualización cada 10s</div>
</div>

<script>
const TOKEN_KEY = 'ocabra_gw_token';
const USER_KEY  = 'ocabra_gw_user';
const GPU_LABEL = { 0: 'RTX 3060 12G', 1: 'RTX 3090 24G' };

const SERVICE_DESC = {
  hunyuan:  'Text or image to 3D — genera meshes 3D texturizados con Hunyuan3D-2.1.',
  comfyui:  'Pipeline de generación de imágenes y vídeo basado en nodos. Compatible con SD, FLUX y workflows personalizados.',
  a1111:    'Automatic1111 Stable Diffusion WebUI — generación de imágenes con control detallado.',
  acestep:  'Generación de música desde texto o audio de referencia. Produce pistas completas.',
};

const STATUS_BADGE = {
  active:      { cls: 'badge-active',   dot: 'dot-active',   label: 'Activo'      },
  idle:        { cls: 'badge-idle',     dot: 'dot-idle',     label: 'Inactivo'    },
  disabled:    { cls: 'badge-disabled', dot: 'dot-disabled', label: 'Desactivado' },
  unreachable: { cls: 'badge-unreach',  dot: 'dot-unreach',  label: 'Offline'     },
  restarting:  { cls: 'badge-starting', dot: 'dot-starting', label: 'Reiniciando' },
  unknown:     { cls: 'badge-idle',     dot: 'dot-idle',     label: 'Desconocido' },
};

// ── Auth helpers ─────────────────────────────────────────────────────────────

function getToken() { return localStorage.getItem(TOKEN_KEY); }
function getUser()  {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null'); } catch { return null; }
}

function isTokenValid(token) {
  if (!token) return false;
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return false;
    const payload = JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/')));
    return payload.exp && payload.exp * 1000 > Date.now() + 60_000; // 1 min buffer
  } catch { return false; }
}

function showLogin() {
  document.getElementById('login-overlay').classList.add('active');
  document.getElementById('main').classList.remove('active');
}

function showMain(user) {
  document.getElementById('login-overlay').classList.remove('active');
  document.getElementById('main').classList.add('active');
  document.getElementById('user-chip').textContent = user?.username || '';
}

async function doLogin() {
  const btn = document.getElementById('login-btn');
  const errEl = document.getElementById('login-error');
  const username = document.getElementById('login-user').value.trim();
  const password = document.getElementById('login-pass').value;
  if (!username || !password) { errEl.textContent = 'Introduce usuario y contraseña.'; return; }

  btn.disabled = true; btn.textContent = 'Entrando…'; errEl.textContent = '';
  try {
    const r = await fetch('/_gw/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await r.json();
    if (!r.ok) { errEl.textContent = data.detail || 'Credenciales incorrectas.'; return; }
    localStorage.setItem(TOKEN_KEY, data.access_token);
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    showMain(data.user);
    refresh();
  } catch (e) {
    errEl.textContent = 'Error de red: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Entrar';
  }
}

function doLogout() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  showLogin();
}

// Enter key on password field
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('login-pass').addEventListener('keydown', e => {
    if (e.key === 'Enter') doLogin();
  });
  document.getElementById('login-user').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('login-pass').focus();
  });
});

// ── Service rendering ─────────────────────────────────────────────────────────

function badge(status) {
  const b = STATUS_BADGE[status] || STATUS_BADGE.unknown;
  return `<span class="badge ${b.cls}"><span class="dot ${b.dot}"></span>${b.label}</span>`;
}
function gpuTag(gpu) {
  if (gpu == null) return '';
  return `<span class="tag">GPU ${gpu}: ${GPU_LABEL[gpu] || 'GPU '+gpu}</span>`;
}
function idleTag(sec) {
  if (!sec) return '';
  return `<span class="tag">Idle: ${Math.round(sec / 60)}m</span>`;
}
function vramTag(mb) {
  if (mb == null) return '';
  return `<span class="tag" title="VRAM usada">💾 ${(mb/1024).toFixed(1)} GB</span>`;
}
function gpuUtilTag(pct) {
  if (pct == null) return '';
  return `<span class="tag" title="GPU utilización">⚡ ${Math.round(pct)}%</span>`;
}
function generatingTag(isGenerating, queueDepth) {
  if (!isGenerating) return '';
  const q = queueDepth > 0 ? ` (+${queueDepth})` : '';
  return `<span class="tag" style="background:#14532d;color:#4ade80">🎨 Generando${q}</span>`;
}

function renderCard(s) {
  const alive  = s.service_alive;
  const canStart = !alive && s.enabled;
  const canStop  = alive && s.enabled;
  const uiUrl  = s.ui_url || '';
  const desc   = SERVICE_DESC[s.service_id] || s.service_type || '';

  return `<div class="card" id="card-${s.service_id}">
    <div class="card-header">
      <span class="card-name">${s.display_name}</span>
      ${badge(s.status)}
    </div>
    ${desc ? `<p class="card-desc">${desc}</p>` : ''}
    <div class="meta">
      ${gpuTag(s.preferred_gpu)}
      ${idleTag(s.idle_unload_after_seconds)}
      ${s.active_model_ref ? `<span class="tag">${s.active_model_ref}</span>` : ''}
    </div>
    ${(s.vram_used_mb != null || s.gpu_util_pct != null || s.is_generating) ? `
    <div class="meta">
      ${generatingTag(s.is_generating, s.queue_depth || 0)}
      ${vramTag(s.vram_used_mb)}
      ${gpuUtilTag(s.gpu_util_pct)}
    </div>` : ''}
    <div class="detail">${s.detail || '&nbsp;'}</div>
    <div class="actions">
      ${uiUrl ? `<a href="${uiUrl}" target="_blank" class="btn btn-open">Abrir ↗</a>` : ''}
      <button class="btn btn-primary" onclick="startService('${s.service_id}', this)"
              ${canStart ? '' : 'disabled'}>Iniciar</button>
      <button class="btn btn-danger"  onclick="stopService('${s.service_id}', this)"
              ${canStop ? '' : 'disabled'}>Parar</button>
    </div>
  </div>`;
}

// ── API calls ─────────────────────────────────────────────────────────────────

async function startService(id, btn) {
  btn.disabled = true; btn.textContent = 'Iniciando…';
  await fetch('/_gw/services/' + id + '/start', { method: 'POST' });
  await refresh();
}

async function stopService(id, btn) {
  btn.disabled = true; btn.textContent = 'Parando…';
  await fetch('/_gw/services/' + id + '/unload', { method: 'POST' });
  await refresh();
}

async function refresh() {
  try {
    const r = await fetch('/_gw/services');
    if (r.status === 401) { doLogout(); return; }
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const services = await r.json();
    const grid = document.getElementById('grid');
    grid.innerHTML = services.length
      ? services.map(renderCard).join('')
      : '<p class="loading-msg">No hay servicios configurados.</p>';
    document.getElementById('refresh-hint').textContent =
      'Actualizado: ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('refresh-hint').textContent = 'Error: ' + e.message;
  }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

const token = getToken();
const user  = getUser();
if (isTokenValid(token)) {
  showMain(user);
  refresh();
  setInterval(refresh, 10000);
} else {
  showLogin();
}
</script>
</body>
</html>"""
