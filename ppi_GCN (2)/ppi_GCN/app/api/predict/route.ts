export async function POST() {
  try {
    console.log("[v0] API predict invoked")
    // Try to read exported predictions from public. If missing, respond with 0.
    // In Next.js, we can fetch from public path at runtime.
    const r = await fetch("/data/predictions.json").catch(() => null as any)
    if (r && r.ok) {
      const data = await r.json()
      console.log("[v0] API predict found predictions", Array.isArray(data) ? data.length : 0)
      return Response.json({ count: Array.isArray(data) ? data.length : 0 })
    }
    console.log("[v0] API predict no predictions file found")
    return Response.json({ count: 0 })
  } catch (e: any) {
    console.log("[v0] API predict error", e?.message)
    return new Response("error", { status: 500 })
  }
}
