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
boot().then(() => showView("create"));

// ---------------------------------------------------------------- tabs ----
const VIEWS = ["create", "library", "mashup", "trends", "mixes"];

function showView(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.view === name));
  document.getElementById("view-library").style.display = name === "library" ? "grid" : "none";
  VIEWS.filter(v => v !== "library").forEach(v => {
    document.getElementById("view-" + v).hidden = v !== name;
  });
  if (name === "create") renderCreate();
  if (name === "mashup") renderMashupForm();
  if (name === "trends") renderTrends();
  if (name === "mixes") renderMixes();
}

document.querySelectorAll(".tab").forEach(t =>
  t.addEventListener("click", () => showView(t.dataset.view)));

// -------------------------------------------------------------- mashup ----
function trackOptions(selected) {
  return tracks.map(t =>
    `<option value="${t.id}" ${t.id === selected ? "selected" : ""}>` +
    `${esc(t.artist ? t.artist + " - " : "")}${esc(t.title)}` +
    `${t.bpm ? ` [${t.bpm} BPM ${t.camelot}]` : ""}</option>`).join("");
}

function renderMashupForm() {
  const el = document.getElementById("view-mashup");
  if (tracks.length < 2) {
    el.innerHTML = `<p class="muted">Need at least two tracks in MUSIC_DIR to build a mashup.</p>`;
    return;
  }
  el.innerHTML = `
    <h2>Build a mashup</h2>
    <p class="muted">Vocal of A layered over the instrumental stems of B, beat-matched and
      key-corrected. First run analyses and separates stems, so it takes a minute.</p>
    <div class="row">
      <div class="field"><label>A — vocal</label>
        <select id="ma">${trackOptions(tracks[0].id)}</select></div>
      <div class="field"><label>B — instrumental</label>
        <select id="mb">${trackOptions(tracks[1] && tracks[1].id)}</select></div>
      <div class="field"><label>Bars</label>
        <input type="number" id="mbars" value="32" min="4" max="256" step="4"></div>
      <div class="field"><label>Outro</label>
        <select id="mtr">
          <option value="">default</option>
          <option value="bass_swap">bass swap</option>
          <option value="filter_sweep">filter sweep</option>
          <option value="echo_out">echo out</option>
          <option value="reverb_tail">reverb tail</option>
        </select></div>
      <button id="mgo">Build mashup</button>
      <button id="mpreview" class="ghost">Preview match only</button>
    </div>
    <div id="mout"></div>`;
  document.getElementById("mgo").onclick = buildMashup;
  document.getElementById("mpreview").onclick = previewMatch;
}

function termBar(name, score, weight, contribution) {
  return `<div class="card">
    <div class="k">${name} <span style="float:right">w ${weight.toFixed(2)}</span></div>
    <div class="v">${(score * 100).toFixed(0)}%</div>
    <div class="bar"><i style="width:${Math.max(0, Math.min(100, score * 100))}%"></i></div>
    <div class="muted">contributes ${contribution.toFixed(3)}</div>
  </div>`;
}

function renderBreakdown(bd) {
  const t = bd.terms, c = bd.contributions;
  return `<div class="terms">
      ${termBar("Tempo", t.tempo.score, t.tempo.weight, c.tempo)}
      ${termBar("Key", t.key.score, t.key.weight, c.key)}
      ${termBar("Chroma", t.chroma.score, t.chroma.weight, c.chroma)}
      ${termBar("Energy", t.energy.score, t.energy.weight, c.energy)}
    </div>
    <h3>Why this pair</h3>
    <div class="reasons">${bd.reasons.map(esc).join("\n")}</div>`;
}

async function previewMatch() {
  const a = document.getElementById("ma").value, b = document.getElementById("mb").value;
  const out = document.getElementById("mout");
  out.innerHTML = `<p class="muted"><span class="spin">◐</span> analysing…</p>`;
  try {
    const r = await api(`/api/match?a=${a}&b=${b}&top=5`);
    if (!r.pairs.length) {
      out.innerHTML = `<p class="err">No compatible section pairs.
        ${esc(r.a.title)} is ${r.a.bpm} BPM, ${esc(r.b.title)} is ${r.b.bpm} BPM —
        outside the tempo tolerance in config.yaml.</p>`;
      return;
    }
    out.innerHTML = `<h3>Top ${r.pairs.length} section pairs</h3>` + r.pairs.map((p, i) => `
      <div class="mix">
        <b>#${i + 1} — score ${p.score.toFixed(3)}</b>
        <div class="muted">${p.section_a.label} @${p.section_a.start}s →
          ${p.section_b.label} @${p.section_b.start}s</div>
        ${i === 0 ? renderBreakdown(p.breakdown) : ""}
      </div>`).join("");
  } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

async function buildMashup() {
  const btn = document.getElementById("mgo"), out = document.getElementById("mout");
  const body = {
    track_a: document.getElementById("ma").value,
    track_b: document.getElementById("mb").value,
    bars: +document.getElementById("mbars").value,
    transition: document.getElementById("mtr").value || null,
  };
  btn.disabled = true; btn.textContent = "rendering…";
  out.innerHTML = `<p class="muted"><span class="spin">◐</span>
    analysing, separating stems, stretching and mixing…</p>`;
  try {
    const r = await api("/api/mashup", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    out.innerHTML = `
      <h3>${esc(r.name)} — ${fmt(r.duration)}</h3>
      <audio controls src="${r.audio_url}"></audio>
      ${r.match ? renderBreakdown(r.match.breakdown) : ""}
      <h3>Render log</h3>
      <div class="reasons">${r.log.map(esc).join("\n")}</div>`;
  } catch (e) {
    out.innerHTML = `<p class="err">${esc(e.message)}</p>`;
  } finally { btn.disabled = false; btn.textContent = "Build mashup"; }
}

// --------------------------------------------------------------- mixes ----
async function renderMixes() {
  const el = document.getElementById("view-mixes");
  el.innerHTML = `<p class="muted">loading…</p>`;
  try {
    const r = await api("/api/mixes");
    el.innerHTML = `<h2>Mixes</h2>` + (r.mixes.length
      ? r.mixes.map(m => `
          <div class="mix">
            <b>${esc(m.title)}</b>
            <div class="muted">${new Date(m.created_at * 1000).toLocaleString()} ·
              ${fmt(m.duration)} · ${esc(m.options.mode || "mix")}
              ${m.options.bars ? "· " + m.options.bars + " bars" : ""}</div>
            <audio controls src="${m.audio_url}"></audio>
          </div>`).join("")
      : `<p class="muted">No mixes yet — build one from the Mashup tab.</p>`);
  } catch (e) { el.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

// -------------------------------------------------------------- create ----
function renderCreate() {
  const el = document.getElementById("view-create");
  if (el.dataset.ready) return;
  el.dataset.ready = "1";
  el.innerHTML = `
    <h2>Create a mix</h2>
    <p class="muted">Pulls the trending list, picks a set weighted by rank, orders it
      for energy flow and renders it. Tracks that are trending but missing from your
      library are listed below so you know what to add.</p>
    <div class="row">
      <div class="field"><label>Tracks from</label>
        <select id="c-source">
          <option value="trends" selected>trending charts</option>
          <option value="library">my library</option>
        </select></div>
      <div class="field"><label>Length</label>
        <select id="c-length">
          <option value="15">15 minutes</option>
          <option value="30">30 minutes</option>
          <option value="60">60 minutes</option>
        </select></div>
      <div class="field"><label>Israel / global</label>
        <select id="c-ratio">
          <option value="0.7" selected>70 / 30 (default)</option>
          <option value="1.0">100% Israel</option>
          <option value="0.5">50 / 50</option>
          <option value="0.3">30 / 70</option>
          <option value="0.0">100% global</option>
        </select></div>
      <div class="field"><label>Hype</label>
        <select id="c-hype">
          <option value="off">off</option>
          <option value="light" selected>light</option>
          <option value="heavy">heavy</option>
        </select></div>
      <div class="field"><label>Blend</label>
        <select id="c-blend">
          <option value="classic" selected>classic DJ</option>
          <option value="mashup">mashup</option>
        </select></div>
      <div class="field"><label>Effects</label>
        <select id="c-fx" title="Production effects applied to transitions">
          <option value="clean">clean</option>
          <option value="club" selected>club — pump, builds, delays</option>
          <option value="atmospheric">atmospheric — + pads &amp; wash</option>
        </select></div>
      <button id="c-go">Create</button>
      <button id="c-analyze" class="ghost"
        title="Pre-separate stems so later mixes render fast">Analyze library</button>
      <button id="c-fetch" class="ghost"
        title="Download Creative Commons music into MUSIC_DIR">Get free music</button>
    </div>
    <div id="c-out"></div>`;
  document.getElementById("c-go").onclick = startCreate;
  document.getElementById("c-analyze").onclick = startAnalyze;
  document.getElementById("c-fetch").onclick = startFetchMusic;
}

function followJob(jobId, out, onDone) {
  out.innerHTML = `<div class="progress"><i style="width:0%"></i></div>
                   <div class="stage">starting…</div>
                   <button id="c-cancel" class="ghost">Cancel</button>`;
  const bar = out.querySelector(".progress > i");
  const stage = out.querySelector(".stage");
  document.getElementById("c-cancel").onclick = () =>
    fetch("/api/jobs/" + jobId + "/cancel", { method: "POST" });

  const es = new EventSource("/api/jobs/" + jobId + "/events");
  es.onmessage = (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch (err) { return; }
    if (typeof d.progress === "number") bar.style.width = (d.progress * 100).toFixed(1) + "%";
    if (d.stage) {
      stage.textContent = d.stage +
        (d.track ? " — " + d.track : "") +
        (d.index ? " (" + d.index + "/" + d.total + ")" : "");
    }
    if (d.status && ["done", "failed", "cancelled"].includes(d.status)) {
      es.close();
      if (d.status === "done") onDone(d.result);
      else out.innerHTML = `<p class="err">${esc(d.error || d.status)}</p>`;
    }
  };
  es.onerror = () => { es.close(); stage.textContent = "connection lost"; };
}

async function startCreate() {
  const out = document.getElementById("c-out");
  const body = {
    source: document.getElementById("c-source").value,
    length_minutes: +document.getElementById("c-length").value,
    israel_ratio: +document.getElementById("c-ratio").value,
    hype: document.getElementById("c-hype").value,
    blend: document.getElementById("c-blend").value,
    fx: document.getElementById("c-fx").value,
  };
  try {
    const job = await api("/api/create", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    followJob(job.job_id, out, renderCreateResult);
  } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

async function startAnalyze() {
  const out = document.getElementById("c-out");
  try {
    const job = await api("/api/library/analyze", { method: "POST" });
    followJob(job.job_id, out, (r) => {
      out.innerHTML = `<p>Analyzed ${r.analyzed} of ${r.total} tracks.` +
        (r.failed.length ? ` <span class="err">${r.failed.length} failed.</span>` : "") +
        `</p>`;
    });
  } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

function missingList(items) {
  return `<div class="missing"><ul>${items.map(m => `<li>
      <b>#${m.rank}</b> ${esc(m.artist)} — ${esc(m.title)}
      <span class="pill ${m.region === "IL" ? "il" : ""}">${m.region}</span>
      ${m.closest_local
        ? `<span class="muted"> · closest: ${esc(m.closest_local)} (${m.closest_score})</span>`
        : ""}
    </li>`).join("")}</ul></div>`;
}

function renderCreateResult(r) {
  const out = document.getElementById("c-out");
  const t = r.tracklist;
  const rows = (t.events || []).filter(e => e.kind === "track_in");
  out.innerHTML = `
    <h3>${esc(r.title)} — ${fmt(r.duration)}</h3>
    <audio controls src="${r.audio_url}"></audio>
    <p class="muted">${t.tracks.length} tracks · ${t.target_bpm} BPM · ${esc(t.mode)} ·
      matched ${r.trends.resolved}/${r.trends.total} trending
      (${(r.trends.coverage * 100).toFixed(0)}% coverage)</p>
    <h3>Tracklist</h3>
    <table class="tl">${rows.map(e => `<tr>
      <td>${e.timecode}</td><td>${esc(e.label)}</td>
      <td class="muted">${e.bpm || ""} ${e.key || ""}</td></tr>`).join("")}</table>
    ${r.missing && r.missing.length ? `
      <h3>Missing from your library <span class="pill warn">${r.missing.length}</span></h3>
      <p class="muted">Trending but not found locally — your shopping list.</p>
      ${missingList(r.missing)}` : ""}
    <h3>Render log</h3>
    <div class="reasons">${(r.log || []).map(esc).join("\n")}</div>`;
}

async function startFetchMusic() {
  const out = document.getElementById("c-out");
  if (!confirm("Download 15 Creative Commons tracks from the Internet Archive " +
               "into MUSIC_DIR? Licences are recorded in LICENCES.txt.")) return;
  try {
    const job = await api("/api/library/fetch-cc?count=15", { method: "POST" });
    followJob(job.job_id, out, (r) => {
      out.innerHTML = `<p>Downloaded ${r.downloaded} tracks into
        <code>${esc(r.directory)}</code>.</p>
        <div class="missing"><ul>${r.tracks.map(t => `<li>${esc(t.artist)} — ${esc(t.title)}
          <span class="muted"> · ${esc(t.license)}</span></li>`).join("")}</ul></div>
        <p class="muted">Set MUSIC_DIR to that folder in .env and restart, then
          press Create with "my library" selected.</p>`;
    });
  } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}

// ------------------------------------------------------------- trending ---
async function renderTrends() {
  const el = document.getElementById("view-trends");
  el.innerHTML = `<p class="muted">fetching charts…</p>`;
  try {
    const r = await api("/api/trends");
    const providers = r.providers
      .map(p => p.name + ": " + (p.available ? "on" : "no key")).join(" · ");
    el.innerHTML = `
      <h2>Trending</h2>
      <p class="muted">${esc(providers)} — matched ${r.counts.resolved} of ${r.total}
        (${(r.coverage * 100).toFixed(0)}% coverage)</p>
      <h3>In your library <span class="pill">${r.counts.resolved}</span></h3>
      <div class="missing"><ul>${r.resolved.map(x => `<li>
        <b>#${x.rank}</b> ${esc(x.artist)} — ${esc(x.title)}
        <span class="pill ${x.region === "IL" ? "il" : ""}">${x.region}</span>
        <span class="muted"> · ${esc(x.providers.join(", "))} · match ${x.match_score}</span>
        </li>`).join("") || `<li class="muted">none matched yet</li>`}</ul></div>
      <h3>Missing <span class="pill warn">${r.counts.missing}</span></h3>
      ${missingList(r.missing.slice(0, 60))}`;
  } catch (e) { el.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
