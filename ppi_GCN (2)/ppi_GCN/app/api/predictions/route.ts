import { type NextRequest, NextResponse } from "next/server"
import { sql } from "@/lib/db"

type IncomingPrediction = {
  run_id: string
  protein_a: string
  protein_b: string
  probability: number
  drug_targets_overlap?: boolean
}

export async function GET(req: NextRequest) {
  const runId = req.nextUrl.searchParams.get("run_id")
  if (!runId) {
    return NextResponse.json({ error: "run_id required" }, { status: 400 })
  }
  const rows = await sql`
    select id, run_id, protein_a, protein_b, probability, drug_targets_overlap, created_at
    from predictions
    where run_id = ${runId}
    order by probability desc nulls last, id asc
    limit 10000
  `
  return NextResponse.json({ predictions: rows })
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const arr: IncomingPrediction[] = Array.isArray(body) ? body : body?.predictions
    if (!Array.isArray(arr) || arr.length === 0) {
      return NextResponse.json(
        { error: "body must be an array of predictions or {predictions:[...]}" },
        { status: 400 },
      )
    }
    const runId = arr[0]?.run_id
    if (!runId) {
      return NextResponse.json({ error: "run_id is required on each prediction" }, { status: 400 })
    }
    // Insert in batches
    for (let i = 0; i < arr.length; i += 500) {
      const chunk = arr.slice(i, i + 500)
      const values = chunk
        .map(
          (p) =>
            `(${sql.unsafe(p.run_id)}, ${sql.unsafe(p.protein_a)}, ${sql.unsafe(p.protein_b)}, ${p.probability ?? null}, ${p.drug_targets_overlap === true})`,
        )
        .join(",")
      await sql`
        insert into predictions (run_id, protein_a, protein_b, probability, drug_targets_overlap)
        values ${sql(values)}
      `
    }
    return NextResponse.json({ ok: true, inserted: arr.length })
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "failed to insert predictions" }, { status: 500 })
  }
}
