import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { LichessPerformanceCurveSeries, LichessPerformanceResponse } from '../services/api'
import { botVersionLabel } from '../lib/botVersionLabel'
import { botChartColor } from '../lib/botChartColors'
import ChartMethodologyHint from './ChartMethodologyHint'

const PHASE_ORDER = ['opening', 'middlegame', 'endgame'] as const

/** Below this n at a move index, the tail is drawn dotted (unstable mean). */
const CPL_CURVE_LOW_N_CUTOFF = 30

/**
 * Half-width in bot move #: each plotted CPL is the mean of raw bin CPL at
 * move indices [i - radius, i + radius] that exist. Larger = smoother curve.
 */
const CPL_CURVE_SMOOTH_RADIUS = 5

const CPL_SOLID_SUFFIX = '__cplSolid'
const CPL_DOTTED_SUFFIX = '__cplDotted'

function cplCurveLineBaseVersion(dataKey: string): string {
  return dataKey.replace(/__(?:cplSolid|cplDotted)$/, '')
}

function phaseLabel(phase: string) {
  if (phase === 'opening') return 'Opening'
  if (phase === 'middlegame') return 'Middlegame'
  return 'Endgame'
}

/** Centered rolling mean of `avg_cpl` over neighboring move indices (same bot). */
function smoothedCplByMove(s: LichessPerformanceCurveSeries, radius: number): Map<number, number | null> {
  const raw = new Map<number, number | null>()
  for (const p of s.points) raw.set(p.bot_move_index, p.avg_cpl)
  const out = new Map<number, number | null>()
  const moveIndices = [...raw.keys()].sort((a, b) => a - b)
  for (const moveIdx of moveIndices) {
    let sum = 0
    let count = 0
    for (let d = -radius; d <= radius; d++) {
      const v = raw.get(moveIdx + d)
      if (v != null && !Number.isNaN(v)) {
        sum += v
        count += 1
      }
    }
    out.set(moveIdx, count === 0 ? null : sum / count)
  }
  return out
}

/** First bot move index where n drops below cutoff (per series); null if never. */
function firstLowNMoveIndex(points: LichessPerformanceCurveSeries['points'], cutoff: number): number | null {
  let cut: number | null = null
  for (const p of points) {
    if (p.n < cutoff) {
      if (cut === null || p.bot_move_index < cut) cut = p.bot_move_index
    }
  }
  return cut
}

function mergeCplCurveRowsWithLowNTail(
  series: LichessPerformanceCurveSeries[],
  cutoff: number,
  smoothRadius: number,
) {
  const idxSet = new Set<number>()
  for (const s of series) {
    for (const p of s.points) idxSet.add(p.bot_move_index)
  }
  const indices = [...idxSet].sort((a, b) => a - b)

  const smoothedByVersion = new Map<string, Map<number, number | null>>()
  for (const s of series) {
    smoothedByVersion.set(s.bot_version, smoothedCplByMove(s, smoothRadius))
  }

  const cuts = new Map<string, number | null>()
  const minIdxByVersion = new Map<string, number | null>()
  for (const s of series) {
    cuts.set(s.bot_version, firstLowNMoveIndex(s.points, cutoff))
    minIdxByVersion.set(
      s.bot_version,
      s.points.length === 0 ? null : Math.min(...s.points.map(p => p.bot_move_index)),
    )
  }

  return indices.map(bot_move_index => {
    const row: Record<string, number | null> = { bot_move_index }
    for (const s of series) {
      const y = smoothedByVersion.get(s.bot_version)?.get(bot_move_index) ?? null
      const cut = cuts.get(s.bot_version)
      const minIdx = minIdxByVersion.get(s.bot_version)
      const solidKey = `${s.bot_version}${CPL_SOLID_SUFFIX}`
      const dottedKey = `${s.bot_version}${CPL_DOTTED_SUFFIX}`

      if (cut == null) {
        row[solidKey] = y
        row[dottedKey] = null
      } else if (minIdx != null && cut <= minIdx) {
        row[solidKey] = null
        row[dottedKey] = y
      } else {
        row[solidKey] = bot_move_index < cut ? y : null
        row[dottedKey] = bot_move_index >= cut - 1 ? y : null
      }
    }
    return row
  })
}

function buildNLookup(series: LichessPerformanceCurveSeries[]) {
  const map = new Map<number, Record<string, number>>()
  for (const s of series) {
    for (const p of s.points) {
      if (!map.has(p.bot_move_index)) map.set(p.bot_move_index, {})
      map.get(p.bot_move_index)![s.bot_version] = p.n
    }
  }
  return map
}

type PhaseMetric = 'blunder' | 'cpl'

export type LichessMovePerformanceChartsProps = {
  data: LichessPerformanceResponse | undefined
  isLoading: boolean
  isError: boolean
}

export default function LichessMovePerformanceCharts({
  data,
  isLoading,
  isError,
}: LichessMovePerformanceChartsProps) {
  const [phaseMetric, setPhaseMetric] = useState<PhaseMetric>('blunder')

  const curveSeries = data?.curve_series ?? []
  const phaseSeries = data?.phase_series ?? []
  const bounds = data?.phase_boundaries
  const total = data?.total_analyzed_bot_moves ?? 0
  const o = bounds?.opening_bot_moves_end ?? '—'
  const m = bounds?.middlegame_bot_moves_end ?? '—'

  const mergedCpl = useMemo(
    () =>
      mergeCplCurveRowsWithLowNTail(curveSeries, CPL_CURVE_LOW_N_CUTOFF, CPL_CURVE_SMOOTH_RADIUS),
    [curveSeries],
  )
  const nLookup = useMemo(() => buildNLookup(curveSeries), [curveSeries])

  const phaseBarRows = useMemo(() => {
    if (phaseSeries.length === 0) return []
    return PHASE_ORDER.map(phaseRaw => {
      const row: Record<string, number | string> = { phase: phaseLabel(phaseRaw) }
      for (const s of phaseSeries) {
        const ph = s.phases.find(x => x.phase === phaseRaw)
        if (phaseMetric === 'blunder') {
          row[s.bot_version] = ph != null && ph.blunder_rate != null ? Math.round(ph.blunder_rate * 10000) / 100 : 0
        } else {
          row[s.bot_version] = ph?.avg_cpl ?? 0
        }
      }
      return row
    })
  }, [phaseSeries, phaseMetric])

  const smoothSpan = CPL_CURVE_SMOOTH_RADIUS * 2 + 1
  const cplByMoveMethodologySections = [
    {
      title: 'Bot move #',
      detail:
        'The Nth move the bot plays in a game (1, 2, …). Each point is the mean centipawn loss (CPL) on that move, averaged over every analyzed game that reached that move number.',
    },
    {
      title: 'CPL (cp)',
      detail:
        'Non-negative Stockfish centipawn loss on the bot’s move (same definition as phase “Avg CPL”). Higher = worse decisions at that index.',
    },
    {
      title: 'Smoothing',
      detail: `The line plots a rolling mean of raw per-move CPL over up to ${smoothSpan} adjacent bot move numbers (±${CPL_CURVE_SMOOTH_RADIUS} indices where data exists), so short noise averages out. Tooltip matches the line.`,
    },
    {
      title: 'Line style',
      detail: `Solid segments: at least ${CPL_CURVE_LOW_N_CUTOFF} games contributing at that move index. Dotted tail: mean is noisy because sample size is small there.`,
    },
  ]

  const phaseMethodologySections = [
    {
      title: 'Move index',
      detail: 'Uses the same “bot move #” as the CPL-by-move chart above.',
    },
    {
      title: 'Phase cutoffs',
      detail: `Opening: bot moves 1–${o}. Middlegame: through bot move ${m}. Endgame: all later bot moves.`,
    },
    {
      title: phaseMetric === 'blunder' ? 'Blunder rate' : 'Avg CPL',
      detail:
        phaseMetric === 'blunder'
          ? 'Percentage of the bot’s moves in that phase tagged as blunders (Lichess classification).'
          : 'Mean centipawn loss on the bot’s moves within that phase.',
    },
    {
      title: 'Bars',
      detail: 'Each colored segment is one bot version. Compare heights within a phase column.',
    },
  ]

  if (isLoading) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 text-center text-sm text-zinc-500">
        Loading…
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 text-center text-sm text-red-400/90">
        Could not load performance data.
      </div>
    )
  }

  const hasCurve = curveSeries.some(s => s.points.length > 0)
  const hasPhase = phaseBarRows.length > 0 && phaseSeries.length > 0

  return (
    <div className="space-y-6">
      <div className="overflow-visible rounded-lg border border-zinc-800 bg-zinc-900 p-4">
        <div className="mb-3 flex items-center gap-1">
          <h3 className="font-medium text-zinc-200">CPL by move</h3>
          <ChartMethodologyHint
            sections={cplByMoveMethodologySections}
            footnote={`${total.toLocaleString()} analyzed bot moves in this query. Dotted tail: fewer than ${CPL_CURVE_LOW_N_CUTOFF} games contributing at that move index for that bot.`}
          />
        </div>

        {!hasCurve ? (
          <p className="text-sm text-zinc-500 py-4 text-center">No curve data — run Analyze from the header.</p>
        ) : (
          <ResponsiveContainer width="100%" height={280} className="[&_.recharts-wrapper]:overflow-visible">
            <LineChart data={mergedCpl} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis
                dataKey="bot_move_index"
                type="number"
                domain={['dataMin', 'dataMax']}
                stroke="#52525b"
                tick={{ fill: '#71717a', fontSize: 11 }}
                label={{ value: 'Bot move #', position: 'insideBottom', offset: -2, fill: '#52525b', fontSize: 11 }}
              />
              <YAxis
                stroke="#fb923c"
                tick={{ fill: '#71717a', fontSize: 11 }}
                label={{ value: 'Avg CPL (cp)', angle: -90, position: 'insideLeft', fill: '#fb923c', fontSize: 11 }}
              />
              <Tooltip
                allowEscapeViewBox={{ x: true, y: true }}
                filterNull={false}
                wrapperStyle={{ zIndex: 50 }}
                content={({ active, label, payload }) => {
                  if (!active || !payload?.length) return null
                  const move = Number(label)
                  const ns = nLookup.get(move)
                  const raw = payload.filter(p => p.dataKey !== 'bot_move_index' && p.value != null)
                  const byBase = new Map<string, (typeof raw)[number]>()
                  for (const p of raw) {
                    const base = cplCurveLineBaseVersion(String(p.dataKey))
                    const prev = byBase.get(base)
                    if (!prev) {
                      byBase.set(base, p)
                      continue
                    }
                    const prevDotted = String(prev.dataKey).endsWith(CPL_DOTTED_SUFFIX)
                    const curDotted = String(p.dataKey).endsWith(CPL_DOTTED_SUFFIX)
                    if (prevDotted && !curDotted) byBase.set(base, p)
                  }
                  const rows = [...byBase.values()].sort((a, b) =>
                    cplCurveLineBaseVersion(String(a.dataKey)).localeCompare(
                      cplCurveLineBaseVersion(String(b.dataKey)),
                    ),
                  )
                  if (rows.length === 0) return null
                  return (
                    <div className="rounded-md border border-zinc-600 bg-zinc-950 px-3 py-2.5 text-xs shadow-lg">
                      <div className="border-b border-zinc-700/90 pb-2">
                        <div className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
                          Bot move
                        </div>
                        <div className="mt-0.5 font-mono text-base font-semibold tabular-nums text-zinc-100">
                          #{label}
                        </div>
                        <div className="mt-1 text-[10px] leading-tight text-zinc-500">
                          Smoothed mean CPL (±{CPL_CURVE_SMOOTH_RADIUS} moves) · n = games at this index
                        </div>
                      </div>
                      <div className="mt-2 grid grid-cols-[minmax(0,1fr)_auto_auto] items-baseline gap-x-3 gap-y-2 text-[11px]">
                        <span className="text-zinc-500">Bot</span>
                        <span className="justify-self-end font-medium text-zinc-500">CPL</span>
                        <span className="justify-self-end font-medium text-zinc-500">n</span>
                        {rows.map(p => {
                          const key = String(p.dataKey)
                          const base = cplCurveLineBaseVersion(key)
                          const sparse = key.endsWith(CPL_DOTTED_SUFFIX)
                          const n = ns?.[base]
                          return (
                            <div key={key} className="contents">
                              <div className="flex min-w-0 items-center gap-1.5">
                                <span
                                  className="h-2 w-2 shrink-0 rounded-full"
                                  style={{ backgroundColor: botChartColor(base) }}
                                  aria-hidden
                                />
                                <span className="truncate font-medium text-zinc-200">
                                  {botVersionLabel(base)}
                                  {sparse ? (
                                    <span className="ml-1 font-normal text-zinc-500">· sparse n</span>
                                  ) : null}
                                </span>
                              </div>
                              <span className="justify-self-end font-mono tabular-nums text-zinc-100">
                                {Number(p.value).toFixed(1)}
                              </span>
                              <span className="justify-self-end font-mono tabular-nums text-zinc-400">
                                {n != null ? n : '—'}
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} formatter={value => botVersionLabel(String(value))} />
              {curveSeries.map(s => (
                <Line
                  key={`${s.bot_version}-solid`}
                  type="monotone"
                  dataKey={`${s.bot_version}${CPL_SOLID_SUFFIX}`}
                  name={s.bot_version}
                  stroke={botChartColor(s.bot_version)}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              ))}
              {curveSeries.map(s => (
                <Line
                  key={`${s.bot_version}-dotted`}
                  type="monotone"
                  dataKey={`${s.bot_version}${CPL_DOTTED_SUFFIX}`}
                  name={`${s.bot_version}_dotted`}
                  stroke={botChartColor(s.bot_version)}
                  strokeWidth={2}
                  strokeDasharray="5 4"
                  dot={false}
                  connectNulls
                  legendType="none"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="overflow-visible rounded-lg border border-zinc-800 bg-zinc-900 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-1">
            <h3 className="font-medium text-zinc-200">By phase</h3>
            <ChartMethodologyHint sections={phaseMethodologySections} />
          </div>
          <div
            className="flex rounded-md border border-zinc-700 bg-zinc-950 p-0.5 text-xs"
            role="group"
            aria-label="Phase metric"
          >
            <button
              type="button"
              onClick={() => setPhaseMetric('blunder')}
              className={`rounded px-2.5 py-1 transition-colors ${
                phaseMetric === 'blunder'
                  ? 'bg-zinc-700 text-zinc-100'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              Blunders
            </button>
            <button
              type="button"
              onClick={() => setPhaseMetric('cpl')}
              className={`rounded px-2.5 py-1 transition-colors ${
                phaseMetric === 'cpl' ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              Avg CPL
            </button>
          </div>
        </div>

        {!hasPhase ? (
          <p className="text-sm text-zinc-500 py-4 text-center">No phase data.</p>
        ) : (
          <ResponsiveContainer width="100%" height={240} className="[&_.recharts-wrapper]:overflow-visible">
            <BarChart data={phaseBarRows} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="phase" stroke="#52525b" tick={{ fill: '#71717a', fontSize: 11 }} />
              <YAxis
                stroke="#52525b"
                tick={{ fill: '#71717a', fontSize: 11 }}
                tickFormatter={v => (phaseMetric === 'blunder' ? `${v}%` : `${v}`)}
                label={{
                  value: phaseMetric === 'blunder' ? '% blunders' : 'CPL',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#52525b',
                  fontSize: 11,
                }}
              />
              <Tooltip
                allowEscapeViewBox={{ x: true, y: true }}
                filterNull={false}
                wrapperStyle={{ zIndex: 50 }}
                content={({ active, label, payload }) => {
                  if (!active || !payload?.length) return null
                  const phaseName = String(label)
                  const rows = payload
                    .filter(p => p.value != null && p.dataKey !== 'phase' && typeof p.dataKey === 'string')
                    .sort((a, b) => String(a.dataKey).localeCompare(String(b.dataKey)))
                  if (rows.length === 0) return null
                  const metricTitle = phaseMetric === 'blunder' ? 'Blunder rate' : 'Avg CPL'
                  const metricHint =
                    phaseMetric === 'blunder'
                      ? 'Percent of bot moves in this phase tagged as blunders'
                      : 'Mean centipawn loss on bot moves in this phase'
                  return (
                    <div className="rounded-md border border-zinc-600 bg-zinc-950 px-3 py-2.5 text-xs shadow-lg">
                      <div className="border-b border-zinc-700/90 pb-2">
                        <div className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
                          Phase
                        </div>
                        <div className="mt-0.5 text-base font-semibold text-zinc-100">{phaseName}</div>
                        <div className="mt-1 text-[10px] leading-tight text-zinc-500">
                          {metricTitle} — {metricHint}
                        </div>
                      </div>
                      <div className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-4 gap-y-2 text-[11px]">
                        <span className="text-zinc-500">Bot</span>
                        <span className="justify-self-end font-medium text-zinc-500">
                          {phaseMetric === 'blunder' ? '%' : 'CPL'}
                        </span>
                        {rows.map(p => {
                          const version = String(p.dataKey)
                          const v = Number(p.value)
                          const display =
                            phaseMetric === 'blunder' ? `${v.toFixed(2)}%` : v.toFixed(1)
                          return (
                            <div key={version} className="contents">
                              <div className="flex min-w-0 items-center gap-1.5">
                                <span
                                  className="h-2 w-2 shrink-0 rounded-full"
                                  style={{ backgroundColor: botChartColor(version) }}
                                  aria-hidden
                                />
                                <span className="truncate font-medium text-zinc-200">
                                  {botVersionLabel(version)}
                                </span>
                              </div>
                              <span className="justify-self-end font-mono tabular-nums text-zinc-100">
                                {display}
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} formatter={value => botVersionLabel(String(value))} />
              {phaseSeries.map(s => (
                <Bar
                  key={s.bot_version}
                  dataKey={s.bot_version}
                  name={s.bot_version}
                  fill={botChartColor(s.bot_version)}
                  radius={[2, 2, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
