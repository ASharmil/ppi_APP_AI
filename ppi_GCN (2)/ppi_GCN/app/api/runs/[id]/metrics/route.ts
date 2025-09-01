import { type NextRequest, NextResponse } from "next/server"
import { sql } from "@/lib/db"

export async function POST(req: NextRequest, { params }: { params: { id: string } }) {
  try {
    const runId = params.id
    const body = await req.json()
    const status = (body?.status || "completed").toString()
    const metrics = body?.metrics ?? body
    await sql /*sql*/`
      update training_runs
      set status = ${status}, metrics = ${JSON.stringify(metrics)}, updated_at = now()
      where id = ${runId}
    `
    return NextResponse.json({ ok: true })
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "failed to update metrics" }, { status: 500 })
  }
}
