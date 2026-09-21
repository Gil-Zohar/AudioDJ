const $ = (s) => document.querySelector(s);
const PITCH = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
let tracks = [], current = null;

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

async function boot() {
  try {
    const h = await api("/api/health");
    $("#status").textContent = h.music_dir_exists
      ? `${h.music_dir} · beats:${h.beat_tracker} · stems:${h.stem_provider}`
      : `MUSIC_DIR not found: ${h.music_dir} — set it in .env`;
    await loadLibrary();
  } catch (e) { $("#status").innerHTML = `<span class="err">${e.message}</span>`; }
}

async function loadLibrary(refresh) {
  const data = await api("/api/library" + (refresh ? "?refresh=true" : ""));
  tracks = data.tracks;
  $("#list").innerHTML = tracks.length
    ? tracks.map((t, i) => `
        <div class="track" data-i="${i}">
          <div class="t">${esc(t.title)}</div>
          <div class="a">${esc(t.artist || "unknown artist")}</div>
          <div style="margin-top:4px">
            ${t.analyzed ? `<span class="badge">${t.bpm} BPM</span><span class="badge">${t.camelot}</span>` : ""}
          </div>
        </div>`).join("")
    : `<div class="muted" style="padding:14px">No audio found. Point MUSIC_DIR at your library and press refresh.</div>`;
  document.querySelectorAll(".track").forEach((el) =>
    el.addEventListener("click", () => select(+el.dataset.i)));
}

async function select(i) {
  current = tracks[i];
  document.querySelectorAll(".track").forEach((e, j) => e.classList.toggle("active", i === j));
  $("#detail").innerHTML = `<h2>${esc(current.title)}</h2>
    <p class="muted">${esc(current.artist)}</p>
    <button id="an">Analyze</button>
    <button id="anf" class="ghost">Re-analyze</button>
    <button id="st" class="ghost">Separate stems</button>
    <div id="out" class="muted" style="margin-top:16px">not analyzed yet</div>`;
  $("#an").onclick = () => analyze(false);
  $("#anf").onclick = () => analyze(true);
  $("#st").onclick = separate;
  try { render(await api(`/api/tracks/${current.id}/analysis`)); } catch { /* not analyzed */ }
}

async function analyze(force) {
  $("#out").textContent = "analyzing… (first run takes a few seconds per track)";
  try {
    render(await api(`/api/tracks/${current.id}/analyze?force=${!!force}`, { method: "POST" }));
    await loadLibrary();
  } catch (e) { $("#out").innerHTML = `<span class="err">${e.message}</span>`; }
}

async function separate() {
  $("#st").disabled = true; $("#st").textContent = "separating…";
  try {
    const s = await api(`/api/tracks/${current.id}/stems`, { method: "POST" });
    $("#st").textContent = `stems ready (${s.provider})`;
  } catch (e) { $("#st").textContent = "stems failed"; alert(e.message); }
  finally { $("#st").disabled = false; }
}

function render(a) {
  const conf = a.key.confidence;
  const warn = conf < 0.35 ? ` <span class="muted">(low — override below)</span>` : "";
  $("#out").innerHTML = `
    <div class="grid">
      <div class="card"><div class="k">Tempo</div><div class="v">${a.bpm}</div></div>
      <div class="card"><div class="k">Key</div><div class="v">${a.key.name}</div>
        <div class="muted">${a.key.camelot} · conf ${conf}${warn} · ${a.key.source}</div></div>
      <div class="card"><div class="k">Beats</div><div class="v">${a.beats}</div>
        <div class="muted">${a.downbeat_count} bars</div></div>
      <div class="card"><div class="k">Length</div><div class="v">${fmt(a.duration)}</div>
        <div class="muted">tracker: ${a.beat_tracker}</div></div>
    </div>
    <h3>Structure</h3>
    <div class="sections">${a.sections.map(s => {
      const w = ((s.end - s.start) / a.duration * 100).toFixed(2);
      return `<div class="sec" style="width:${w}%;background:var(--${s.label})" title="${s.label} ${fmt(s.start)}–${fmt(s.end)} energy ${s.energy}">${s.label}</div>`;
    }).join("")}</div>
    <h3>Energy</h3>
    <canvas id="energy"></canvas>
    <h3>Override key</h3>
    <p class="muted">Detection is unreliable on Mizrahi/maqam material — pin it by hand if it looks wrong.</p>
    <select id="pc">${PITCH.map((p, i) => `<option value="${i}" ${i === a.key.pitch_class ? "selected" : ""}>${p}</option>`).join("")}</select>
    <select id="md">${["major", "minor"].map(m => `<option ${m === a.key.mode ? "selected" : ""}>${m}</option>`).join("")}</select>
    <button id="setk">Set</button><button id="clrk" class="ghost">Reset to detected</button>
    <h3>Sections</h3>
    <table><tr><th>#</th><th>label</th><th>start</th><th>end</th><th>energy</th></tr>
      ${a.sections.map(s => `<tr><td>${s.index}</td><td>${s.label}</td><td>${fmt(s.start)}</td><td>${fmt(s.end)}</td><td>${s.energy}</td></tr>`).join("")}
    </table>`;
  drawEnergy(a);
  $("#setk").onclick = async () => {
    render(await api(`/api/tracks/${current.id}/key`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pitch_class: +$("#pc").value, mode: $("#md").value }) }));
  };
  $("#clrk").onclick = async () => {
    await api(`/api/tracks/${current.id}/key`, { method: "DELETE" });
    render(await api(`/api/tracks/${current.id}/analysis`));
  };
}

function drawEnergy(a) {
  const c = $("#energy"), dpr = devicePixelRatio || 1;
  c.width = c.clientWidth * dpr; c.height = 90 * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr);
  const w = c.clientWidth, h = 90, v = a.energy_curve;
  if (!v.length) return;
  g.fillStyle = "#171b22"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#4fd1c5"; g.lineWidth = 1.5; g.beginPath();
  v.forEach((y, i) => { const x = i / (v.length - 1) * w, py = h - y * (h - 6) - 3;
    i ? g.lineTo(x, py) : g.moveTo(x, py); });
  g.stroke();
  g.strokeStyle = "#ffffff22";
  a.sections.forEach(s => { const x = s.start / a.duration * w;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); });
}

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
boot();
