// Optional details endpoint if needed later
export async function GET(_: Request, { params }: { params: { accession: string } }) {
  try {
    const r = await fetch(`https://rest.uniprot.org/uniprotkb/${params.accession}.json`)
    if (!r.ok) throw new Error("uniprot error")
    const data = await r.json()
    return Response.json(data)
  } catch (e) {
    return Response.json({}, { status: 404 })
  }
}
