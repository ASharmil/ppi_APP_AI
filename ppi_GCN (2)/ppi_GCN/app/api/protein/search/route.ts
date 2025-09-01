import type { NextRequest } from "next/server"

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const query = searchParams.get("query") || ""
  if (!query) {
    return Response.json({ results: [] })
  }
  try {
    const url = `https://rest.uniprot.org/uniprotkb/search?query=${encodeURIComponent(query)}&fields=accession,protein_name,gene_primary,organism_name&size=10&format=json`
    const r = await fetch(url, { headers: { Accept: "application/json" } })
    if (!r.ok) throw new Error("uniprot error")
    const data = await r.json()
    const results = (data.results || []).map((it: any) => ({
      accession: it.primaryAccession,
      proteinName: it?.proteinDescription?.recommendedName?.fullName?.value || it?.uniProtkbId || "Protein",
      gene: it?.genes?.[0]?.geneName?.value || "",
      organism: it?.organism?.scientificName || "",
    }))
    return Response.json({ results })
  } catch (e) {
    return Response.json({ results: [] })
  }
}
