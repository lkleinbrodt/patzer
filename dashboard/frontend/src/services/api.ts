import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// ── Eval types ──────────────────────────────────────────────────────────────

export interface PlayerRating {
  name: string
  elo: number
  stderr: number | null
  games: number
  wins: number
  losses: number
  draws: number
  is_stockfish: boolean
}

export interface LeaderboardResponse {
  ratings: PlayerRating[]
  total_games: number
  models_evaluated: number
  best_model: string | null
  best_elo: number | null
}

export interface ProgressPoint {
  iter: number
  elo: number
  stderr: number | null
  games: number
}

export interface ProgressSeries {
  version: string
  points: ProgressPoint[]
}

export interface ProgressResponse {
  series: ProgressSeries[]
}

export interface H2HRecord {
  wins: number
  losses: number
  draws: number
  games: number
  win_rate: number | null
}

export interface H2HResponse {
  players: string[]
  matrix: Record<string, Record<string, H2HRecord>>
}

export interface TerminationEntry {
  type: string
  count: number
}

export interface TerminationsResponse {
  distribution: TerminationEntry[]
  total: number
}

// ── Lichess types ────────────────────────────────────────────────────────────

export interface LichessGame {
  id: string
  bot_version: string
  bot_color: string
  opponent: string | null
  opponent_rating: number | null
  bot_rating: number | null
  result: string
  speed: string | null
  opening_name: string | null
  played_at: string | null
  avg_eval_loss_cp: number | null
  blunders: number | null
  analyzed: boolean
}

export interface LichessStatsEntry {
  total_games: number
  wins: number
  losses: number
  draws: number
  win_rate: number | null
  avg_cpl: number | null
  blunder_rate: number | null
  analyzed_games: number
  /** Lichess account used for API export / public profile (e.g. patzer_v2b for patzer_v2). */
  lichess_username: string
  /** Glicko-2 display rating from Lichess public API, if the mode exists on the profile. */
  lichess_bullet_rating: number | null
  lichess_blitz_rating: number | null
}

export interface LichessStatsResponse {
  by_version: Record<string, LichessStatsEntry>
  total_games: number
}

export interface GamesResponse {
  games: LichessGame[]
  total: number
  limit: number
  offset: number
}

export interface LichessPerformanceCurveRow {
  bot_move_index: number
  n: number
  avg_eval_before_cp: number | null
  avg_cpl: number | null
  blunder_rate: number
  mistake_rate: number
  inaccuracy_rate: number
}

export interface LichessPerformancePhaseRow {
  phase: string
  bot_moves: number
  avg_cpl: number | null
  blunder_rate: number
  mistake_rate: number
  inaccuracy_rate: number
}

export interface LichessPerformanceCurveSeries {
  bot_version: string
  points: LichessPerformanceCurveRow[]
}

export interface LichessPerformancePhaseSeries {
  bot_version: string
  phases: LichessPerformancePhaseRow[]
}

export interface LichessPerformanceResponse {
  bot_version: string | null
  total_analyzed_bot_moves: number
  max_bot_move_index: number
  min_per_bin: number
  phase_boundaries: {
    opening_bot_moves_end: number
    middlegame_bot_moves_end: number
  }
  /** One series per bot; each `points` row includes `avg_cpl` (curve chart) and `avg_eval_before_cp` (unused in UI). */
  curve_series: LichessPerformanceCurveSeries[]
  phase_series: LichessPerformancePhaseSeries[]
}

export interface BotSyncStatus {
  status: string
  fetched: number
  last_sync_at: string | null
}

export interface SyncStatusResponse {
  status: string
  lines: string[]
  error: string | null
  bots: Record<string, BotSyncStatus>
}

// ── API functions ────────────────────────────────────────────────────────────

export const evalApi = {
  getLeaderboard: () => api.get<LeaderboardResponse>('/eval/leaderboard'),
  getProgress: () => api.get<ProgressResponse>('/eval/progress'),
  getH2H: () => api.get<H2HResponse>('/eval/h2h'),
  getTerminations: () => api.get<TerminationsResponse>('/eval/terminations'),
}

export const lichessApi = {
  sync: (botVersions?: string[]) =>
    api.post<{ status: string }>('/lichess/sync', { bot_versions: botVersions }),
  analyze: () => api.post<{ status: string }>('/lichess/analyze'),
  getAnalyzeStatus: () => api.get<{ running: boolean }>('/lichess/analyze/status'),
  getSyncStatus: () => api.get<SyncStatusResponse>('/lichess/sync/status'),
  syncStreamUrl: () => '/api/lichess/sync/stream',
  getStats: () => api.get<LichessStatsResponse>('/lichess/stats'),
  getGames: (params?: {
    bot_version?: string
    speed?: string
    result?: string
    limit?: number
    offset?: number
  }) => api.get<GamesResponse>('/lichess/games', { params }),
  getPerformance: (params?: {
    bot_version?: string
    max_bot_move_index?: number
    min_per_bin?: number
    opening_end?: number
    middlegame_end?: number
  }) => api.get<LichessPerformanceResponse>('/lichess/performance', { params }),
}
