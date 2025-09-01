import { type NextRequest, NextResponse } from "next/server"
import { sql } from "@/lib/db"

export async function GET() {
  const rows = await sql /*sql*/`
    select id, name, created_at
    from datasets
    order by created_at desc
    limit 200
  `
  return NextResponse.json({ datasets: rows })
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const name = (body?.name || "").toString().slice(0, 200)
    const csvText = (body?.csvText || "").toString()
    if (!name || !csvText) {
      return NextResponse.json({ error: "name and csvText are required" }, { status: 400 })
    }
    const id = crypto.randomUUID()
    await sql /*sql*/`
      insert into datasets (id, name, csv_text)
      values (${id}, ${name}, ${csvText})
    `
    return NextResponse.json({ id, name })
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "failed to create dataset" }, { status: 500 })
  }
}
