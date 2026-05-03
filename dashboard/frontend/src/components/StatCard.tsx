interface StatCardProps {
  label: string
  value: string | number | null
  sub?: string
  accent?: boolean
}

export default function StatCard({ label, value, sub, accent }: StatCardProps) {
  return (
    <div className={`rounded-lg border p-4 ${accent ? 'border-amber-500/30 bg-amber-500/5' : 'border-zinc-800 bg-zinc-900'}`}>
      <div className="text-xs text-zinc-500 uppercase tracking-wider">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${accent ? 'text-amber-400' : 'text-zinc-100'}`}>
        {value ?? '—'}
      </div>
      {sub && <div className="mt-0.5 text-xs text-zinc-500">{sub}</div>}
    </div>
  )
}
