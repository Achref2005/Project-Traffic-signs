// Proxies the uploaded image to the Python inference service (server/infer_api.py).
// Query parameters (e.g. ?conf=0.5) are forwarded unchanged.
export const runtime = "nodejs";

export async function DELETE(req) {
  const backend = process.env.MODEL_API_URL || "http://127.0.0.1:8001";
  const sid = new URL(req.url).searchParams.get("session") || "";
  try {
    const r = await fetch(`${backend}/session/${encodeURIComponent(sid)}`, { method: "DELETE" });
    return Response.json(await r.json(), { status: r.status });
  } catch (e) {
    return Response.json({ closed: false }, { status: 502 });
  }
}

export async function POST(req) {
  const backend = process.env.MODEL_API_URL || "http://127.0.0.1:8001";
  try {
    const form = await req.formData();
    const file = form.get("image");
    if (!file || typeof file === "string") {
      return Response.json({ error: "no image in request" }, { status: 400 });
    }
    const fd = new FormData();
    fd.append("image", file, file.name || "upload.jpg");
    const qs = new URL(req.url).search; // forward ?conf=... etc.
    const r = await fetch(`${backend}/infer${qs}`, { method: "POST", body: fd });
    const body = await r.json();
    return Response.json(body, { status: r.status });
  } catch (e) {
    return Response.json(
      { error: `model service unreachable at ${backend} — start it with: python server/infer_api.py` },
      { status: 502 }
    );
  }
}
