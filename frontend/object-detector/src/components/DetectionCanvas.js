import { useRef, useEffect } from "react";

export default function DetectionCanvas({ video, detections }) {
  const ref = useRef(null);

  useEffect(() => {
    const c = ref.current;
    if (!c || !video) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    c.width = video.videoWidth || 640;
    c.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, c.width, c.height);

    ctx.lineWidth = 2;
    ctx.font = "14px system-ui, sans-serif";
    detections.forEach(d => {
      const [x1, y1, x2, y2] = d.xyxy;
      ctx.strokeStyle = "#00ff88";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${d.cls} ${(d.conf * 100).toFixed(0)}%`;
      const w = ctx.measureText(label).width + 8, h = 18;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(x1, y1 - h, w, h);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + 4, y1 - 5);
    });
  }, [video, detections]);

  return (
    <canvas
      ref={ref}
      style={{ width: "100%", borderRadius: 12, boxShadow: "0 6px 20px rgba(0,0,0,0.25)" }}
    />
  );
}
