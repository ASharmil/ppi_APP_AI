import { type NextRequest, NextResponse } from "next/server"
import { sql } from "@/lib/db"

export async function GET(_req: NextRequest, { params }: { params: { id: string } }) {
  const id = params.id
  const rows = await sql /*sql*/`
    select csv_text from datasets where id = ${id} limit 1
  `
  if (!rows.length) {
    return new NextResponse("Not Found", { status: 404 })
  }
  const csv = rows[0].csv_text as string
  return new NextResponse(csv, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${id}.csv"`,
    },
  })
}
