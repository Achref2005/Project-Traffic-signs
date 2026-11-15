"use client";

import { useState, useRef, useEffect } from "react";

export default function Page() {
  const [mode, setMode] = useState("image"); // "image" | "video"

  // — image mode —
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  // — shared —
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const [conf, setConf] = useState(0.30);
  const [imgsz, setImgSz] = useState(416);

  // — video (live) mode —
  const [videoName, setVideoName] = useState("");
  const [videoURL, setVideoURL] = useState(null);
  const [live, setLive] = useState(false);
  const [liveDets, setLiveDets] = useState([]);
  const [liveStats, setLiveStats] = useState({ frames: 0, signs: 0, classes: {} });
  const [vstats, setVStats] = useState({ playing: false, w: 0, h: 0, sent: 0, ok: 0, http: null });

  const inputRef = useRef(null);
  const videoRef = useRef(null);
  const overlayRef = useRef(null); // canvas drawn on top of the playing video
  const captureRef = useRef(null); // hidden canvas that grabs frames
  const busyRef = useRef(false);   // one frame request at a time
  const timerRef = useRef(null);
  const confRef = useRef(conf);
  confRef.current = conf;
  const imgszRef = useRef(imgsz);
  imgszRef.current = imgsz;
  const lastRef = useRef([]);   // hold the last detections to stop box flickering
  const missRef = useRef(0);    // consecutive empty frames
  const zeroRef = useRef(0);    // videoWidth===0 watchdog counter

  useEffect(() => () => clearInterval(timerRef.current), []);

  function resetLiveStats() {
    setLiveDets([]);
    setLiveStats({ frames: 0, signs: 0, classes: {} });
    lastRef.current = [];
    missRef.current = 0;
    zeroRef.current = 0;
    resetVStats();
  }

  function resetVStats() {
    setVStats({ playing: false, w: 0, h: 0, sent: 0, ok: 0, http: null });
  }

  function pick(f) {
    if (!f) return;
    if (f.type.startsWith("video/")) return pickVideo(f);
    if (!f.type.startsWith("image/")) return;
    stopLive();
    setError(null); setResult(null);
    if (preview) URL.revokeObjectURL(preview);
    if (videoURL) URL.revokeObjectURL(videoURL);
    setMode("image");
    setVideoName(""); setVideoURL(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    resetLiveStats();
  }

  function pickVideo(f) {
    stopLive();
    setError(null); setResult(null);
    if (preview) URL.revokeObjectURL(preview);
    if (videoURL) URL.revokeObjectURL(videoURL);
    setMode("video");
    setFile(null); setPreview(null);
    setVideoName(f.name);
    setVideoURL(URL.createObjectURL(f));
    resetLiveStats();
  }

  function switchMode(m) {
    if (m === mode) return;
    stopLive();
    setError(null); setResult(null);
    if (preview) URL.revokeObjectURL(preview);
    if (videoURL) URL.revokeObjectURL(videoURL);
    setFile(null); setPreview(null); setVideoName(""); setVideoURL(null);
    resetLiveStats();
    setImgSz(m === "video" ? 960 : 416); // dashcam signs are small → scan harder
    setMode(m);
  }

  function clearAll() {
    stopLive();
    if (mode === "image") {
      if (preview) URL.revokeObjectURL(preview);
      setFile(null); setPreview(null);
    } else {
      if (videoURL) URL.revokeObjectURL(videoURL);
      setVideoName(""); setVideoURL(null);
    }
    setResult(null); setError(null);
    resetLiveStats();
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("image", file);
      const r = await fetch(`/api/infer?conf=${confRef.current}&imgsz=${imgszRef.current}`, { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || data.detail || `HTTP ${r.status}`);
      setResult(data);
    } catch (e) {
      setError(e.message + " — make sure the model server is running:  python server/infer_api.py");
    } finally {
      setLoading(false);
    }
  }

  // — video live loop -------------------------------------------------------
  function stopLive() {
    clearInterval(timerRef.current);
    timerRef.current = null;
    busyRef.current = false;
    setLive(false);
  }

  function toggleLive() {
    if (live) return stopLive();
    const v = videoRef.current;
    if (!v || !videoURL) return;
    setError(null);
    resetVStats();
    if (v.paused || v.ended) v.play().catch(() => {});
    setLive(true);
    clearInterval(timerRef.current);
    timerRef.current = setInterval(grabAndSend, 110); // ~9 analysed fps
    grabAndSend();
  }

  function grabAndSend() {
    const v = videoRef.current, c = captureRef.current;
    if (!v || !c) return;
    if (v.paused || v.ended) { setVStats((s) => ({ ...s, playing: false })); return; }
    if (!v.videoWidth) {
      zeroRef.current += 1;                                  // video never decoded
      if (zeroRef.current > 20) {
        setError("The video never decoded (videoWidth = 0) — the browser can't play this file/codec. Convert it to mp4 H.264 (VLC/HandBrake) and try again.");
        stopLive();
      }
      return;
    }
    zeroRef.current = 0;
    // keep uploaded frames light: cap the long side at 1600 px
    const scale = Math.min(1, 1600 / v.videoWidth);
    const w = Math.round(v.videoWidth * scale) || v.videoWidth;
    const h = Math.round(v.videoHeight * scale) || v.videoHeight;
    c.width = w; c.height = h;
    c.getContext("2d", { alpha: false }).drawImage(v, 0, 0, w, h);
    busyRef.current = true;
    c.toBlob((blob) => {
      if (!blob) { busyRef.current = false; return; }
      const fd = new FormData();
      fd.append("image", new File([blob], "frame.jpg", { type: "image/jpeg" }));
      fetch(`/api/infer?conf=${confRef.current}&imgsz=${imgszRef.current}`, { method: "POST", body: fd })
        .then(async (r) => {
          const st = r.status;
          const data = await r.json().catch(() => ({}));
          setVStats((s) => ({
            ...s, playing: !v.paused, w: v.videoWidth, h: v.videoHeight,
            http: st, sent: s.sent + 1, ok: s.ok + (st === 200 ? 1 : 0),
          }));
          if (data.error || data.detail) throw new Error(data.error || data.detail);
          const dets = data.detections || [];
          if (dets.length) { lastRef.current = dets; missRef.current = 0; }
          else { missRef.current += 1; }
          const shown = dets.length ? dets
            : (missRef.current < 3 ? lastRef.current : []); // hold boxes briefly on miss
          setLiveDets(shown);
          setLiveStats((s) => {
            const classes = { ...s.classes };
            dets.forEach((d) => { classes[d.yolo_class] = (classes[d.yolo_class] || 0) + 1; });
            return { frames: s.frames + 1, signs: s.signs + dets.length, classes };
          });
          drawOverlay(w, h, shown);
        })
        .catch((e) => {
          setError(e.message + " — make sure the model server is running:  python server/infer_api.py");
          stopLive();
        })
        .finally(() => { busyRef.current = false; });
    }, "image/jpeg", 0.6);
  }

  function drawOverlay(w, h, detections) {
    const cv = overlayRef.current;
    if (!cv) return;
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.lineJoin = "round";
    ctx.textBaseline = "top";
    const lw = Math.max(2, Math.round(w / 320));
    const font = Math.max(12, Math.round(w / 56));
    const pad = 3, th = font + pad * 2;
    ctx.font = `${font}px "Segoe UI", sans-serif`;
    for (const d of detections) {
      const [x1, y1, x2, y2] = d.box;
      ctx.lineWidth = lw;
      ctx.strokeStyle = "#ffc400";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const l1 = `${d.yolo_class} ${Math.round(d.yolo_conf * 100)}%`;
      const ly1 = Math.max(0, y1 - th - 2);
      ctx.fillStyle = "rgba(30, 20, 0, .78)";
      ctx.fillRect(x1, ly1, ctx.measureText(l1).width + pad * 2, th);
      ctx.fillStyle = "#ffd54a";
      ctx.fillText(l1, x1 + pad, ly1 + pad);
      if (d.cnn_class) {
        const l2 = d.cnn_class;
        const ly2 = Math.min(h - th - 1, y2 + 2);
        ctx.fillStyle = "rgba(0, 35, 12, .78)";
        ctx.fillRect(x1, ly2, ctx.measureText(l2).width + pad * 2, th);
        ctx.fillStyle = "#7dff8a";
        ctx.fillText(l2, x1 + pad, ly2 + pad);
      }
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDrag(false);
    pick(e.dataTransfer.files?.[0]);
  }

  return (
    <div className="page">
      <div className="hero">
        <h1>🚦 Traffic Sign Recognition</h1>
        <p>
          YOLOv8n detects the panels · GTSRB CNN names each one.<br />
          Upload a photo — or drop dashcam footage and watch the models work live.
        </p>
      </div>

      <div className="modebar">
        <button className={`modebtn ${mode === "image" ? "on" : ""}`} onClick={() => switchMode("image")}>🖼 Photo</button>
        <button className={`modebtn ${mode === "video" ? "on" : ""}`} onClick={() => switchMode("video")}>🎥 Dashcam video</button>
      </div>

      <div
        className={`card ${drag ? "dropzone drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        {mode === "image" ? (
          <>
            <div className="dropzone" onClick={() => inputRef.current?.click()}>
              <p style={{ margin: 0 }}>
                {file ? <>Selected: <strong>{file.name}</strong></> : <>Drag &amp; drop an image here, or <strong>browse</strong></>}
              </p>
              {preview && <img className="preview" src={preview} alt="selected" />}
            </div>
            <div className="row">
              <button className="btn" disabled={!file || loading} onClick={analyze}>
                {loading ? "Analyzing…" : "🔍 Detect signs"}
              </button>
              {file && (
                <button className="btn" style={{ background: "#22304d" }} disabled={loading} onClick={clearAll}>
                  Clear
                </button>
              )}
            </div>
          </>
        ) : (
          <>
            <div className="dropzone" onClick={() => inputRef.current?.click()}>
              <p style={{ margin: 0 }}>
                {videoName
                  ? <>Selected: <strong>{videoName}</strong> — press ▶ start for live detection</>
                  : <>Drag &amp; drop a dashcam video (mp4/avi/webm), or <strong>browse</strong></>}
              </p>
            </div>
            {videoURL && (
              <div className="video-wrap">
                <video
                  ref={videoRef}
                  src={videoURL}
                  controls
                  muted
                  playsInline
                  onEnded={stopLive}
                  onError={() => {
                    stopLive();
                    setError("The browser can't play this video file (unsupported codec) — convert to mp4/H.264 (VLC or HandBrake) and try again.");
                  }}
                />
                <canvas ref={overlayRef} />
                <canvas ref={captureRef} hidden /> {/* frame grabber — non-visual on purpose */}
              </div>
            )}
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={mode === "image" ? "image/*" : "video/*"}
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
      </div>

      <div className="confrow">
        <label htmlFor="conf">Detection confidence</label>
        <input
          id="conf"
          type="range"
          min="0.10"
          max="0.90"
          step="0.05"
          value={conf}
          onChange={(e) => setConf(parseFloat(e.target.value))}
        />
        <strong>{conf.toFixed(2)}</strong>
      </div>

      <div className="confrow">
        <span>Sign scale in frame</span>
        {[416, 640, 960, 1280].map((s) => (
          <button key={s} className={`sizebtn ${imgsz === s ? "on" : ""}`} onClick={() => setImgSz(s)}>
            {s}
          </button>
        ))}
        <span className="hint">{mode === "video" ? "small/far signs → higher" : "close-up signs → 416"}</span>
      </div>

      {mode === "video" && videoURL && (
        <div className="row">
          <button className="btn" onClick={toggleLive}>
            {live ? "⏹ Stop live detection" : "▶ Start live detection"}
          </button>
          <button className="btn" style={{ background: "#22304d" }} onClick={clearAll}>
            Clear
          </button>
        </div>
      )}

      {mode === "video" && videoURL && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="livehead">
            {live && <span className="dot" />}
            {live ? "Live detection running" : "Live detection idle — press ▶ start"}
          </div>
          <p style={{ color: "var(--muted)", fontSize: 12.5, margin: "8px 0 0" }}>
            {vstats.playing ? "▶ playing" : "⏸ not playing"}
            {vstats.w ? ` ${vstats.w}×${vstats.h}` : " (no frames decoded)"}
            {" · "}requests sent <strong style={{ color: "var(--text)" }}>{vstats.sent}</strong>
            {vstats.http != null && <>{" · "}last HTTP <strong style={{ color: "var(--text)" }}>{vstats.http}</strong></>}
            {!live && !vstats.sent && <> — start the video and press ▶ start</>}
          </p>
          {liveStats.signs > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
              <span className="chip orange">🎞 frames {liveStats.frames}</span>
              <span className="chip orange">signs {liveStats.signs}</span>
              {Object.entries(liveStats.classes).map(([cls, n]) => (
                <span key={cls} className="chip">{cls} × {n}</span>
              ))}
            </div>
          )}
          {liveDets.length === 0 ? (
            <p style={{ textAlign: "center", color: "var(--muted)", margin: "10px 0 0" }}>
              {live ? "No sign on the current frame…" : "Detections appear here while the video plays."}
            </p>
          ) : (
            <div className="dets" style={{ marginTop: 10 }}>
              {liveDets.map((d, i) => (
                <div className="det" key={i}>
                  <span className="chip">YOLO · {d.yolo_class}</span>
                  {d.cnn_class && <span className="chip green">CNN · {d.cnn_class}</span>}
                  <span className="bar">det {Math.round(d.yolo_conf * 100)}%{d.cnn_class ? `  ·  cls ${Math.round(d.cnn_conf * 100)}%` : ""}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {loading && (
        <div className="card" style={{ marginTop: 20, textAlign: "center" }}>
          <div className="spinner" />
          Running YOLOv8 + CNN…
        </div>
      )}

      {error && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="error">{error}</div>
        </div>
      )}

      {result && mode === "image" && (
        <div className="card result" style={{ marginTop: 20 }}>
          {result.detections.length === 0
            ? <p style={{ textAlign: "center", color: "var(--muted)" }}>No traffic sign detected — try an image with signs closer to the camera.</p>
            : <p style={{ margin: 0, color: "var(--muted)" }}>{result.detections.length} panel(s) detected.</p>}
          <img src={result.annotated} alt="annotated result" />
          <div className="dets">
            {result.detections.map((d, i) => (
              <div className="det" key={i}>
                <span className="chip">YOLO · {d.yolo_class}</span>
                {d.cnn_class && <span className="chip green">CNN · {d.cnn_class}</span>}
                <span className="bar">det {Math.round(d.yolo_conf * 100)}%{d.cnn_class ? `  ·  cls ${Math.round(d.cnn_conf * 100)}%` : ""}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="tip">
        API service: <code>python server/infer_api.py</code> before using the UI.<br />
        Endpoints — POST <code>/infer</code> (image, conf) · GET <code>/health</code><br />
        Video mode samples ~9 frames/s while the file plays and draws YOLO/CNN boxes live on the overlay.<br />
        Small/far signs barely visible at 416 px — raise <em>Sign scale</em> (imgsz) and lower confidence.
      </p>
    </div>
  );
}
