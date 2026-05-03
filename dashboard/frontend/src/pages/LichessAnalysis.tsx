import { useQuery } from '@tanstack/react-query'
import { lichessApi } from '../services/api'
import { botVersionLabel } from '../lib/botVersionLabel'
import { BOT_VERSION_CHART_COLORS } from '../lib/botChartColors'
import LichessMovePerformanceCharts from '../components/LichessMovePerformanceCharts'

export default function LichessAnalysis() {
  const { data: stats } = useQuery({
    queryKey: ['lichess', 'stats'],
    queryFn: () => lichessApi.getStats().then(r => r.data),
  })

  const perfQuery = useQuery({
    queryKey: ['lichess', 'performance', ''],
    queryFn: () => lichessApi.getPerformance({}).then(r => r.data),
  })

  const versions = Object.keys(stats?.by_version ?? {}).sort()
  const totalGames = stats?.total_games ?? 0

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-zinc-100">Lichess</h2>
        <p className="text-sm text-zinc-500 mt-0.5">
          {totalGames > 0 ? `${totalGames.toLocaleString()} games` : 'No games yet'}
        </p>
      </div>

      <section className="space-y-3">
        <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wide">Leaderboard</h3>
        {versions.length === 0 ? (
          <p className="text-sm text-zinc-500">Sync from the header, then refresh.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-900">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-left text-xs text-zinc-500 uppercase tracking-wide">
                  <th className="px-4 py-2 font-medium">Bot</th>
                  <th className="px-4 py-2 font-medium text-right" title="Lichess bullet rating (public API)">
                    Bullet
                  </th>
                  <th className="px-4 py-2 font-medium text-right" title="Lichess blitz rating (public API)">
                    Blitz
                  </th>
                  <th className="px-4 py-2 font-medium text-right">Games</th>
                  <th className="px-4 py-2 font-medium text-right">W-L-D</th>
                  <th className="px-4 py-2 font-medium text-right">Win</th>
                  <th className="px-4 py-2 font-medium text-right">CPL</th>
                  <th className="px-4 py-2 font-medium text-right">Blunder</th>
                  <th className="px-4 py-2 font-medium text-right">Analyzed</th>
                </tr>
              </thead>
              <tbody>
                {versions.map(version => {
                  const s = stats!.by_version[version]
                  const wr = s.win_rate != null ? `${Math.round(s.win_rate * 100)}%` : '—'
                  const cpl = s.avg_cpl != null ? s.avg_cpl.toFixed(1) : '—'
                  const br = s.blunder_rate != null ? `${(s.blunder_rate * 100).toFixed(1)}%` : '—'
                  const bullet = s.lichess_bullet_rating
                  const blitz = s.lichess_blitz_rating
                  const profileTitle = `${s.lichess_username} on Lichess`
                  return (
                    <tr key={version} className="border-b border-zinc-800/80 last:border-0">
                      <td className="px-4 py-2.5">
                        <a
                          href={`https://lichess.org/@/${encodeURIComponent(s.lichess_username)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-mono text-xs font-medium underline-offset-2 hover:underline"
                          style={{ color: BOT_VERSION_CHART_COLORS[version] ?? '#94a3b8' }}
                          title={`${version} → @${s.lichess_username}`}
                        >
                          {botVersionLabel(version)}
                        </a>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300 tabular-nums">
                        {bullet != null ? (
                          <a
                            href={`https://lichess.org/@/${encodeURIComponent(s.lichess_username)}/perf/bullet`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-emerald-400/90 hover:text-emerald-300 underline-offset-2 hover:underline"
                            title={profileTitle}
                          >
                            {bullet}
                          </a>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300 tabular-nums">
                        {blitz != null ? (
                          <a
                            href={`https://lichess.org/@/${encodeURIComponent(s.lichess_username)}/perf/blitz`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-emerald-400/90 hover:text-emerald-300 underline-offset-2 hover:underline"
                            title={profileTitle}
                          >
                            {blitz}
                          </a>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300 tabular-nums">{s.total_games}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-400 tabular-nums text-xs">
                        {s.wins}-{s.losses}-{s.draws}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300 tabular-nums">{wr}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300 tabular-nums">{cpl}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300 tabular-nums">{br}</td>
                      <td className="px-4 py-2.5 text-right text-zinc-400 tabular-nums">
                        {s.analyzed_games}/{s.total_games}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="space-y-3">
        <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wide">Game performance</h3>
        <LichessMovePerformanceCharts
          data={perfQuery.data}
          isLoading={perfQuery.isLoading}
          isError={perfQuery.isError}
        />
      </section>
    </div>
  )
}
