import { type NextRequest, NextResponse } from "next/server"
import { sql } from "@/lib/db"

export async function GET(req: NextRequest) {
  const datasetId = req.nextUrl.searchParams.get("dataset_id")
  const rows = datasetId
    ? await sql /*sql*/`
        select id, dataset_id, status, notes, metrics, created_at, updated_at
        from training_runs
        where dataset_id = ${datasetId}
        order by created_at desc
        limit 200
      `
    : await sql /*sql*/`
        select id, dataset_id, status, notes, metrics, created_at, updated_at
        from training_runs
        order by created_at desc
        limit 200
      `
  return NextResponse.json({ runs: rows })
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const datasetId = (body?.dataset_id || "").toString()
    const notes = (body?.notes || "").toString().slice(0, 500)
    if (!datasetId) {
      return NextResponse.json({ error: "dataset_id is required" }, { status: 400 })
    }
    const id = crypto.randomUUID()
    await sql /*sql*/`
      insert into training_runs (id, dataset_id, status, notes)
      values (${id}, ${datasetId}, 'created', ${notes})
    `
    // Provide a ready-to-run example command for local training:
    const datasetURL = `/api/datasets/${datasetId}/download`
    const hint = {
      run_id: id,
      fetch_dataset: `curl -sSL "${datasetURL}" -o data/train_${datasetId}.csv`,
      train: `python -m scripts.train_gcn_v1 --csv data/train_${datasetId}.csv --save-dir artifacts/${id}`,
      infer_hint: `python -m scripts.infer_gcn_v1 --csv data/dynamic_eval.csv --artifacts artifacts/${id}/artifacts.json --model artifacts/${id}/gcn_lp.pt --out outputs/${id}_scores.csv`,
      upload_metrics: `curl -X POST /api/runs/${id}/metrics -H "Content-Type: application/json" -d "@artifacts/${id}/metrics.json"`,
      upload_predictions: `curl -X POST /api/predictions -H "Content-Type: application/json" -d "@outputs/${id}_scores.json"`,
    }
    return NextResponse.json({ id, dataset_id: datasetId, notes, hint })
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "failed to create run" }, { status: 500 })
  }
}
