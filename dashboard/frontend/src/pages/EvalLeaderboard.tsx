import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, BarChart, Bar,
} from 'recharts'
import { evalApi, type PlayerRating } from '../services/api'
import StatCard from '../components/StatCard'
import SortableTable, { type Column } from '../components/SortableTable'

const VERSION_COLORS: Record<string, string> = {
  patzer_v1: '#f59e0b',
  patzer_v2: '#10b981',
  patzer_v3: '#60a5fa',
  patzer_v4: '#e879f9',
}

const TERMINATION_COLORS = ['#f59e0b', '#60a5fa', '#34d399', '#f87171', '#a78bfa']

export default function EvalLeaderboard() {
  const { data: lb, isLoading: lbLoading } = useQuery({
    queryKey: ['eval', 'leaderboard'],
    queryFn: () => evalApi.getLeaderboard().then(r => r.data),
  })

  const { data: progress } = useQuery({
    queryKey: ['eval', 'progress'],
    queryFn: () => evalApi.getProgress().then(r => r.data),
  })

  const { data: h2h } = useQuery({
    queryKey: ['eval', 'h2h'],
    queryFn: () => evalApi.getH2H().then(r => r.data),
  })

  const { data: terminations } = useQuery({
    queryKey: ['eval', 'terminations'],
    queryFn: () => evalApi.getTerminations().then(r => r.data),
  })

  if (lbLoading) {
    return <div className="text-zinc-500 py-12 text-center">Loading…</div>
  }

  const leaderboardCols: Column<PlayerRating & Record<string, unknown>>[] = [
    {
      key: 'rank',
      label: '#',
      render: row => (
        <span className="text-zinc-500 font-mono text-xs">{(row.rank as number) + 1}</span>
      ),
    },
    {
      key: 'name',
      label: 'Player',
      render: row => (
        <span className="font-mono text-sm text-zinc-100">
          {row.name}
        </span>
      ),
    },
    {
      key: 'elo',
      label: 'Elo',
      sortable: true,
      render: row => (
        <span className="font-semibold text-amber-400">
          {row.elo}
          {row.stderr != null && (
            <span className="text-zinc-500 font-normal text-xs ml-1">±{row.stderr}</span>
          )}
        </span>
      ),
    },
    { key: 'games', label: 'Games', sortable: true },
    {
      key: 'wld',
      label: 'W-L-D',
      render: row => (
        <span className="font-mono text-xs">
          <span className="text-green-400">{row.wins}</span>
          <span className="text-zinc-600">-</span>
          <span className="text-red-400">{row.losses}</span>
          <span className="text-zinc-600">-</span>
          <span className="text-zinc-400">{row.draws}</span>
        </span>
      ),
    },
    {
      key: 'win_rate',
      label: 'Win%',
      sortable: true,
      render: row => {
        const total = row.wins + row.losses + row.draws
        const rate = total ? Math.round((100 * row.wins) / total) : null
        return rate != null ? `${rate}%` : '—'
      },
    },
  ]

  const tableRows = (lb?.ratings ?? []).map((r, i) => ({
    ...r,
    rank: i,
    wld: '',
    win_rate: r.wins + r.losses + r.draws > 0 ? r.wins / (r.wins + r.losses + r.draws) : 0,
  }))

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-zinc-100 mb-1">Local Eval Leaderboard</h2>
        <p className="text-sm text-zinc-500">Bradley-Terry MLE ratings computed from {lb?.total_games?.toLocaleString()} simulated games</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard label="Total Games" value={lb?.total_games?.toLocaleString() ?? '—'} />
        <StatCard label="Models Evaluated" value={lb?.models_evaluated ?? '—'} />
        <StatCard label="Best Model" value={lb?.best_model ?? '—'} accent />
        <StatCard label="Best Elo" value={lb?.best_elo ?? '—'} accent />
      </div>

      {/* Leaderboard table */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
        <div className="px-4 py-3 border-b border-zinc-800">
          <h3 className="font-medium text-zinc-200">Rankings</h3>
        </div>
        <SortableTable
          columns={leaderboardCols as Column<Record<string, unknown>>[]}
          rows={tableRows as Record<string, unknown>[]}
          rowKey={r => String(r.name)}
        />
      </div>

      {/* Elo progress chart */}
      {progress && progress.series.length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <h3 className="font-medium text-zinc-200 mb-4">Elo vs Training Iterations</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis
                dataKey="iter"
                type="number"
                domain={['auto', 'auto']}
                tickFormatter={v => `${v}k`}
                stroke="#52525b"
                tick={{ fill: '#71717a', fontSize: 11 }}
                label={{ value: 'Training steps (k)', position: 'insideBottom', offset: -2, fill: '#52525b', fontSize: 11 }}
              />
              <YAxis
                stroke="#52525b"
                tick={{ fill: '#71717a', fontSize: 11 }}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 6 }}
                labelStyle={{ color: '#a1a1aa' }}
                labelFormatter={v => `${v}k steps`}
                formatter={(val) => [`${val} Elo`]}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {progress.series.map(s => (
                <Line
                  key={s.version}
                  data={s.points}
                  dataKey="elo"
                  name={s.version}
                  stroke={VERSION_COLORS[s.version] ?? '#94a3b8'}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Termination distribution */}
        {terminations && terminations.distribution.length > 0 && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="font-medium text-zinc-200 mb-4">Game Terminations</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={terminations.distribution} layout="vertical" margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
                <XAxis type="number" stroke="#52525b" tick={{ fill: '#71717a', fontSize: 11 }} />
                <YAxis
                  dataKey="type"
                  type="category"
                  width={130}
                  stroke="#52525b"
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  tickFormatter={v => v.replace(/_/g, ' ')}
                />
                <Tooltip
                  contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 6 }}
                  labelStyle={{ color: '#a1a1aa' }}
                />
                <Bar dataKey="count" radius={[0, 3, 3, 0]}>
                  {terminations.distribution.map((_, i) => (
                    <Cell key={i} fill={TERMINATION_COLORS[i % TERMINATION_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* H2H matrix */}
        {h2h && h2h.players.length > 0 && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="font-medium text-zinc-200 mb-3">Head-to-Head Win Rates</h3>
            <p className="text-xs text-zinc-500 mb-3">Row player win rate vs column player (as white)</p>
            <div className="overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr>
                    <th className="text-zinc-500 font-normal pb-1 pr-2 text-left">vs →</th>
                    {h2h.players.map(p => (
                      <th key={p} className="text-zinc-400 pb-1 px-1 font-mono whitespace-nowrap" title={p}>
                        {p.replace('patzer_', '')}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {h2h.players.map(src => (
                    <tr key={src}>
                      <td className="font-mono text-zinc-400 pr-2 py-0.5 whitespace-nowrap">
                        {src.replace('patzer_', '')}
                      </td>
                      {h2h.players.map(dst => {
                        if (src === dst) return <td key={dst} className="px-1 py-0.5 text-center text-zinc-700">—</td>
                        const rec = h2h.matrix[src]?.[dst]
                        if (!rec) return <td key={dst} className="px-1 py-0.5 text-center text-zinc-700">—</td>
                        const rate = rec.win_rate
                        const color = rate == null ? '' : rate > 0.6 ? 'text-green-400' : rate < 0.4 ? 'text-red-400' : 'text-zinc-300'
                        return (
                          <td key={dst} className={`px-1 py-0.5 text-center font-mono ${color}`} title={`${rec.wins}W-${rec.losses}L-${rec.draws}D (${rec.games} games)`}>
                            {rate != null ? `${Math.round(rate * 100)}%` : '—'}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
