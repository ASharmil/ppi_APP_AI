import type { NextRequest } from "next/server"

// Simple ChEMBL-based lookup: find targets by UniProt accession or name, then list a few molecules.
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const query = searchParams.get("query") || ""
  if (!query) return Response.json([])

  try {
    // 1) Find targets by UniProt accession or name
    const tUrl = `https://www.ebi.ac.uk/chembl/api/data/target.json?target_components__accession=${encodeURIComponent(query)}&limit=5`
    const tRes = await fetch(tUrl)
    let targets = []
    if (tRes.ok) {
      const tData = await tRes.json()
      targets = tData.targets || tData?.page ? tData.page : []
    }

    // fallback: search by text if direct accession not found
    if (!targets.length) {
      const sUrl = `https://www.ebi.ac.uk/chembl/api/data/target/search.json?q=${encodeURIComponent(query)}&limit=5`
      const sRes = await fetch(sUrl)
      if (sRes.ok) {
        const sData = await sRes.json()
        targets = sData.page || []
      }
    }

    const targetIds = (targets || []).map((t: any) => t.target_chembl_id).filter(Boolean)
    if (!targetIds.length) return Response.json([])

    // 2) Fetch a few approved molecules linked to these targets (simplified)
    // Note: ChEMBL complex joins are simplified here for demo.
    const unique: Record<string, any> = {}
    for (const tid of targetIds.slice(0, 3)) {
      const mUrl = `https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=10&target_chembl_id=${tid}`
      const mRes = await fetch(mUrl)
      if (mRes.ok) {
        const mData = await mRes.json()
        const list = mData.molecules || mData.page || []
        for (const m of list) {
          unique[m.chembl_id || m.molecule_chembl_id || m.pref_name || Math.random().toString()] = m
        }
      }
    }
    return Response.json(Object.values(unique).slice(0, 20))
  } catch (e) {
    return Response.json([])
  }
}
