"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ProteinViewer } from "@/components/protein-viewer"
import { DataUpload } from "@/components/data-upload"
import { ArduinoPanel } from "@/components/arduino-panel"
import { logAction } from "@/lib/blockchain"
import { cn } from "@/lib/utils"
import { DatasetPanel } from "@/components/dataset-panel" // add dataset dashboard
import { RunsPanel } from "@/components/runs-panel" // add runs dashboard

type ProteinHit = {
  accession: string
  proteinName: string
  gene?: string
  organism?: string
}

type Prediction = {
  query_uniprot: string
  predicted_partner_uniprot: string
  score: number
}

async function searchProteins(query: string): Promise<ProteinHit[]> {
  if (!query) return []
  const res = await fetch(`/api/protein/search?query=${encodeURIComponent(query)}`)
  if (!res.ok) return []
  const data = await res.json()
  return data.results as ProteinHit[]
}

async function fetchBestPdbForAccession(accession: string): Promise<string | null> {
  const res = await fetch(`/api/structure/by-uniprot/${accession}`)
  if (!res.ok) return null
  const data = await res.json()
  return data.bestPdbId || null
}

async function fetchPredictions(accession: string): Promise<Prediction[]> {
  // For now, use sample predictions file produced by Python script export
  const res = await fetch(`/data/predictions.sample.json`)
  if (!res.ok) return []
  const all = (await res.json()) as Prediction[]
  return all.filter((p) => p.query_uniprot === accession)
}

async function fetchDrugs(accessionOrName: string) {
  const res = await fetch(`/api/drug/search?query=${encodeURIComponent(accessionOrName)}`)
  if (!res.ok) return []
  return res.json()
}

export default function HomePage() {
  const [query, setQuery] = useState("")
  const [loading, setLoading] = useState(false)
  const [hits, setHits] = useState<ProteinHit[]>([])
  const [selected, setSelected] = useState<ProteinHit | null>(null)
  const [pdbId, setPdbId] = useState<string | null>(null)
  const [preds, setPreds] = useState<Prediction[]>([])
  const [drugItems, setDrugItems] = useState<any[]>([])
  const [viewerKey, setViewerKey] = useState(0) // force rerender viewer

  const onSearch = async () => {
    setLoading(true)
    const results = await searchProteins(query)
    setHits(results)
    setLoading(false)
    logAction("protein_search", { query, count: results.length }).catch(() => {})
  }

  const onSelectProtein = async (hit: ProteinHit) => {
    setSelected(hit)
    setPdbId(null)
    setPreds([])
    setDrugItems([])
    setViewerKey((v) => v + 1)
    // parallel fetches
    const [bestPdb, predictions, drugs] = await Promise.all([
      fetchBestPdbForAccession(hit.accession),
      fetchPredictions(hit.accession),
      fetchDrugs(hit.accession || hit.proteinName || hit.gene || ""),
    ])
    setPdbId(bestPdb)
    setPreds(predictions)
    setDrugItems(drugs || [])
    logAction("protein_select", { accession: hit.accession, pdbId: bestPdb, preds: predictions.length }).catch(() => {})
  }

  const brandTitle = useMemo(() => "PPI-GCN Studio", [])

  return (
    <main className="min-h-dvh bg-white text-black font-sans">
      <div className="mx-auto max-w-6xl p-4 md:p-6">
        <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-semibold text-balance">{brandTitle}</h1>
            <p className="text-sm text-neutral-600">
              GCN-based Protein-Protein Interaction prediction with 3D ribbon viewer, drug insights, and secure logging
            </p>
          </div>
          <div className="flex gap-2">
            <ArduinoPanel />
          </div>
        </header>

        <section className="mt-6 grid gap-4 md:grid-cols-[1.2fr,2fr]">
          <Card>
            <CardHeader>
              <CardTitle>Search Proteins</CardTitle>
              <CardDescription>Type a protein name, UniProt accession, or gene symbol</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <div className="flex gap-2">
                <Input
                  placeholder="e.g. TP53 or P04637"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onSearch()}
                />
                <Button onClick={onSearch} disabled={loading}>
                  {loading ? "Searching..." : "Search"}
                </Button>
              </div>
              <div className="flex flex-col divide-y">
                {hits.map((h) => (
                  <button
                    key={h.accession}
                    onClick={() => onSelectProtein(h)}
                    className={cn(
                      "text-left py-2 hover:bg-neutral-50 rounded px-2",
                      selected?.accession === h.accession && "bg-teal-50",
                    )}
                  >
                    <div className="font-medium">
                      {h.proteinName} <span className="text-xs text-neutral-500">({h.accession})</span>
                    </div>
                    <div className="text-xs text-neutral-600">
                      {h.gene} • {h.organism}
                    </div>
                  </button>
                ))}
                {hits.length === 0 && !loading && <div className="text-sm text-neutral-500">No results yet</div>}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>3D Ribbon Viewer</CardTitle>
              <CardDescription>Rotate, zoom, and inspect interactions</CardDescription>
            </CardHeader>
            <CardContent>
              <ProteinViewer key={viewerKey} pdbId={pdbId} />
            </CardContent>
          </Card>
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Predicted Interactions</CardTitle>
              <CardDescription>GCN predictions for selected protein</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {preds.length === 0 && <div className="text-sm text-neutral-500">No predictions yet</div>}
              {preds.map((p, i) => (
                <div key={i} className="flex items-center justify-between rounded border p-2">
                  <div>
                    <div className="text-sm">
                      Partner UniProt: <span className="font-mono">{p.predicted_partner_uniprot}</span>
                    </div>
                    <div className="text-xs text-neutral-600">Score: {p.score.toFixed(3)}</div>
                  </div>
                  <Button
                    variant="secondary"
                    onClick={async () => {
                      const best = await fetchBestPdbForAccession(p.predicted_partner_uniprot)
                      if (best) {
                        setPdbId(best)
                        setViewerKey((v) => v + 1)
                      }
                      logAction("open_predicted_partner", { partner: p.predicted_partner_uniprot, pdbId: best }).catch(
                        () => {},
                      )
                    }}
                  >
                    View structure
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Drug Insights</CardTitle>
              <CardDescription>Real-time from public drug databases</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {drugItems?.length === 0 && <div className="text-sm text-neutral-500">No drug data</div>}
              {drugItems?.map((d: any, i: number) => (
                <div key={i} className="rounded border p-2">
                  <div className="font-medium">{d.pref_name || d.molecule_name || d.name}</div>
                  <div className="text-xs text-neutral-600">
                    {d.max_phase != null && <>Max phase: {d.max_phase} • </>}
                    {d.molecule_type && <>Type: {d.molecule_type}</>}
                    {d.first_approval && <> • First approval: {d.first_approval}</>}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </section>

        <section className="mt-6">
          <Tabs defaultValue="dataset">
            <TabsList>
              <TabsTrigger value="dataset">Dataset (CSV)</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>
            <TabsContent value="dataset" className="mt-3">
              <DataUpload
                onPreview={(rows) => {
                  logAction("dataset_preview", { rows: rows.length }).catch(() => {})
                }}
              />
              <div className="mt-4">
                <DatasetPanel />
              </div>
            </TabsContent>
            <TabsContent value="advanced" className="mt-3">
              <div className="text-sm text-neutral-600">
                • Training is performed by the Python scripts in /scripts. Export predictions JSON to
                public/data/predictions.sample.json to surface here.
                <br />• Blockchain logging runs automatically when you search/select/predict. No explicit nav item is
                shown.
                <br />• Arduino panel (top right) lets you connect and stream interaction signals.
              </div>
            </TabsContent>
          </Tabs>
        </section>

        <section className="mt-6">
          <RunsPanel />
        </section>
      </div>
    </main>
  )
}
