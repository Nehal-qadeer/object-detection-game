// src/App.js
import React, { useEffect, useRef, useState } from "react";
import { detectFrame, fetchClasses, fetchQuizOptions } from "./api";
import DetectionCanvas from "./components/DetectionCanvas";

function speak(text) {
  try {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  } catch {}
}

export default function App() {
  const videoRef = useRef(null);
  const initRef = useRef(false); // prevent double init in CRA dev

  const [detections, setDetections] = useState([]);
  const [best, setBest] = useState(null); // {cls, conf} from backend
  const [running, setRunning] = useState(false);
  const [latency, setLatency] = useState(null);
  const [classes, setClasses] = useState([]);
  const [subtitle, setSubtitle] = useState("");

  // Quiz state
  const [quizActive, setQuizActive] = useState(false);
  const [quizOptions, setQuizOptions] = useState([]); // string[]
  const [quizCorrectIndex, setQuizCorrectIndex] = useState(-1);
  const [quizSelected, setQuizSelected] = useState(-1);

  // helpers for subtitles
  const showSub = (text, ms = 2500) => {
    setSubtitle(text);
    setTimeout(() => setSubtitle(""), ms);
  };

  // Load class names once
  useEffect(() => {
    (async () => {
      try {
        const data = await fetchClasses();
        setClasses(data.classes || []);
      } catch (e) {
        console.warn("Failed to fetch classes:", e);
      }
    })();
  }, []);

  // Setup webcam once (avoid double-run in StrictMode)
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    let stream;

    const setup = async () => {
      try {
        const v = videoRef.current;
        if (!v) return;

        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        });

        v.srcObject = stream;

        const onLoaded = () => {
          v.play().catch(() => {}); // ignore autoplay warnings
        };
        v.addEventListener("loadedmetadata", onLoaded, { once: true });
      } catch (err) {
        console.error("camera error:", err);
      }
    };

    setup();

    return () => {
      try {
        const v = videoRef.current;
        if (v) {
          v.pause();
          v.srcObject = null;
        }
        if (stream) {
          stream.getTracks().forEach((t) => t.stop());
        }
      } catch {}
    };
  }, []);

  // Capture frame -> detect
  const captureAndDetect = async () => {
    const v = videoRef.current;
    if (!v) return;

    const vw = v.videoWidth || 0,
      vh = v.videoHeight || 0;
    if (vw === 0 || vh === 0) {
      return; // try again on next loop tick
    }

    const canvas = document.createElement("canvas");
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((res) =>
      canvas.toBlob((b) => res(b || new Blob()), "image/jpeg", 0.85)
    );

    const t0 = performance.now();
    const resp = await detectFrame(blob);
    const t1 = performance.now();

    setLatency(resp.latency_ms ?? Math.round(t1 - t0));
    setDetections(resp.detections || []);
    setBest(resp.best || null);
  };

  // Polling loop
  useEffect(() => {
    let id;
    if (running) {
      const loop = async () => {
        await captureAndDetect();
        id = window.setTimeout(loop, 150);
      };
      loop();
    }
    return () => clearTimeout(id);
  }, [running]); // eslint-disable-line react-hooks/exhaustive-deps

  // Speak top detection
  const speakTop = () => {
    const top = best?.cls || detections[0]?.cls;
    if (!top) return;
    const phrase = `This is ${top}.`;
    showSub(phrase);
    speak(phrase);
  };

  // Start quiz using best detection's class
  const startQuiz = async () => {
    const top = best?.cls || detections[0]?.cls;
    if (!top) {
      showSub("No detection yet. Click Single Shot or Start Detection.");
      return;
    }
    try {
      const data = await fetchQuizOptions(top, 4); // {options, correct_index}
      setQuizOptions(data.options || []);
      setQuizCorrectIndex(typeof data.correct_index === "number" ? data.correct_index : -1);
      setQuizSelected(-1);
      setQuizActive(true);
      // announce the image prompt, like your desktop app
      showSub("What object is this?");
      speak("What object is this?");
    } catch (e) {
      console.error("quiz error:", e);
      showSub("Could not start quiz.");
    }
  };

  // Handle answer click
  const chooseAnswer = (idx) => {
    if (!quizActive) return;
    setQuizSelected(idx);

    const correctName = quizOptions[quizCorrectIndex];
    const chosenName = quizOptions[idx];

    if (idx === quizCorrectIndex) {
      const msg = `Yes, this is ${correctName}.`;
      showSub(msg);
      speak(msg);
    } else {
      const msg = `No, it is not ${chosenName}, it is ${correctName}.`;
      showSub(msg);
      speak(msg);
    }

    // auto-close quiz after a moment
    setTimeout(() => setQuizActive(false), 1500);
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0b0f13", color: "#eee", padding: 24 }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 12 }}>
          Thesis Object Detector (Web)
        </h1>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{
              width: "100%",
              borderRadius: 12,
              boxShadow: "0 6px 20px rgba(0,0,0,0.25)",
              background: "#000",
            }}
          />
          <DetectionCanvas video={videoRef.current} detections={detections} />
        </div>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            alignItems: "center",
            marginTop: 14,
          }}
        >
          <button
            onClick={() => setRunning((v) => !v)}
            style={{
              padding: "10px 16px",
              borderRadius: 10,
              background: "#10b981",
              border: "none",
              color: "#08110a",
              fontWeight: 700,
            }}
          >
            {running ? "Stop Detection" : "Start Detection"}
          </button>

          <button
            onClick={captureAndDetect}
            style={{
              padding: "10px 16px",
              borderRadius: 10,
              background: "#3b82f6",
              border: "none",
              color: "#eef5ff",
              fontWeight: 700,
            }}
          >
            Single Shot
          </button>

          <button
            onClick={speakTop}
            style={{
              padding: "10px 16px",
              borderRadius: 10,
              background: "#f59e0b",
              border: "none",
              color: "#1a1200",
              fontWeight: 700,
            }}
          >
            🔊 Speak Top
          </button>

          <button
            onClick={startQuiz}
            disabled={!best && !detections.length}
            style={{
              padding: "10px 16px",
              borderRadius: 10,
              background: !best && !detections.length ? "#6b7280" : "#a78bfa",
              border: "none",
              color: "#0b0820",
              fontWeight: 700,
            }}
            title={!best && !detections.length ? "Take a detection first" : ""}
          >
            ▶ Start Quiz
          </button>

          <span style={{ opacity: 0.8 }}>
            {latency !== null ? `Latency: ${latency} ms` : "Latency: –"}
          </span>
          {best?.cls && (
            <span style={{ opacity: 0.8, marginLeft: 8 }}>
              Top: <strong>{best.cls}</strong>
            </span>
          )}
        </div>

        {/* Quiz panel */}
        {quizActive && (
          <div
            style={{
              marginTop: 16,
              padding: 16,
              borderRadius: 12,
              background: "#111827",
              boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
            }}
          >
            <div style={{ fontSize: 18, marginBottom: 10, color: "#cbd5e1" }}>
              What object is this?
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              {quizOptions.map((opt, idx) => {
                const isSel = idx === quizSelected;
                const isCorrect = idx === quizCorrectIndex;
                const bg = !isSel ? "#1f2937" : isCorrect ? "#10b981" : "#ef4444";
                const fg = !isSel ? "#e5e7eb" : isCorrect ? "#03261b" : "#220606";
                return (
                  <button
                    key={idx}
                    onClick={() => chooseAnswer(idx)}
                    style={{
                      padding: "12px 14px",
                      borderRadius: 10,
                      border: "none",
                      textAlign: "left",
                      background: bg,
                      color: fg,
                      fontWeight: 700,
                    }}
                  >
                    {opt}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Subtitle banner */}
        {subtitle && (
          <div
            style={{
              position: "fixed",
              left: 0,
              right: 0,
              bottom: 0,
              padding: "12px 18px",
              background: "rgba(0,0,0,0.55)",
              backdropFilter: "blur(2px)",
              color: "#fff",
              fontSize: 18,
            }}
          >
            {subtitle}
          </div>
        )}

        <p style={{ opacity: 0.7, fontSize: 13, marginTop: 12 }}>
          Classes: {classes.length ? classes.join(", ") : "loading…"}
        </p>
      </div>
    </div>
  );
}
