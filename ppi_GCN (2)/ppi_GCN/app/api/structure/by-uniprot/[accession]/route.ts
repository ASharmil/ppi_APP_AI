export async function GET(_: Request, { params }: { params: { accession: string } }) {
  const { accession } = params
  try {
    // RCSB search for entries mapped to UniProt accession
    const payload = {
      query: {
        type: "terminal",
        service: "text",
        parameters: {
          attribute: "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
          operator: "exact_match",
          value: accession,
        },
      },
      return_type: "entry",
      request_options: { paginate: { rows: 25 } },
    }
    const r = await fetch(
      "https://search.rcsb.org/rcsbsearch/v2/query?json=" + encodeURIComponent(JSON.stringify(payload)),
    )
    if (!r.ok) throw new Error("rcsb error")
    const data = await r.json()
    const ids: string[] = (data.result_set || []).map((x: any) => x.identifier)
    // Pick the first as "best" (could refine by resolution/length)
    const bestPdbId = ids[0] || null
    return Response.json({ bestPdbId, ids })
  } catch (e) {
    return Response.json({ bestPdbId: null, ids: [] })
  }
}
