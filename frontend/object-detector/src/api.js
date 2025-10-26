// src/api.js — use CRA proxy so requests go to the backend without CORS issues.
// In package.json (frontend), set:  "proxy": "http://127.0.0.1:8000"

export async function fetchClasses() {
  const res = await fetch('/classes');
  if (!res.ok) throw new Error('classes request failed');
  return res.json(); // { classes: [...] }
}

export async function detectFrame(imageBlob) {
  const form = new FormData();
  form.append('image', imageBlob, 'frame.jpg');
  let res;
  try {
    res = await fetch("/detect", { method: "POST", body: form });
  } catch (e) {
    console.error("NETWORK error calling /detect:", e);
    throw new Error("detect request failed (network)");
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    console.error("DETECT error status:", res.status, "body:", text);
    throw new Error(`detect request failed (status ${res.status})`);
  }
  return res.json(); // { detections: [...], best, latency_ms }
}
// quiz helper

export async function fetchQuizOptions(correct, k = 4) {
  const params = new URLSearchParams({ correct, k: String(k) });
  const res = await fetch(`/quiz/options?${params.toString()}`);
  if (!res.ok) throw new Error("quiz options request failed");
  return res.json(); // { options: string[], correct_index: number }
}
