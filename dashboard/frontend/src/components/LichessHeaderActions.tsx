import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { lichessApi, type SyncStatusResponse } from '../services/api'

/** Compact Sync + Analyze for the main header (Lichess routes only). */
export default function LichessHeaderActions() {
  const queryClient = useQueryClient()
  const [progress, setProgress] = useState<SyncStatusResponse | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzeNote, setAnalyzeNote] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const analyzePollRef = useRef<number | null>(null)
  const clearDoneRef = useRef<number | null>(null)

  const closeStream = useCallback(() => {
    esRef.current?.close()
    esRef.current = null
    setStreaming(false)
  }, [])

  const startStream = useCallback(() => {
    const es = new EventSource(lichessApi.syncStreamUrl())
    esRef.current = es
    es.onmessage = ev => {
      try {
        const data = JSON.parse(ev.data) as SyncStatusResponse
        setProgress(data)
        if (data.status === 'done' || data.status === 'error') {
          closeStream()
          queryClient.invalidateQueries({ queryKey: ['lichess'] })
        }
      } catch { /* ignore */ }
    }
    es.onerror = () => closeStream()
  }, [closeStream, queryClient])

  const runSync = useCallback(async () => {
    setAnalyzeNote(null)
    if (clearDoneRef.current != null) {
      window.clearTimeout(clearDoneRef.current)
      clearDoneRef.current = null
    }
    try {
      await lichessApi.sync()
    } catch (e: unknown) {
      const err = e as { response?: { status: number } }
      if (err.response?.status === 409) {
        setStreaming(true)
        startStream()
        return
      }
      return
    }
    setProgress({ status: 'running', lines: [], error: null, bots: {} })
    setStreaming(true)
    startStream()
  }, [startStream])

  useEffect(() => () => {
    if (analyzePollRef.current != null) window.clearInterval(analyzePollRef.current)
    if (clearDoneRef.current != null) window.clearTimeout(clearDoneRef.current)
  }, [])

  useEffect(() => {
    if (progress?.status !== 'done') return
    if (clearDoneRef.current != null) window.clearTimeout(clearDoneRef.current)
    clearDoneRef.current = window.setTimeout(() => {
      setProgress(null)
      clearDoneRef.current = null
    }, 5000)
    return () => {
      if (clearDoneRef.current != null) {
        window.clearTimeout(clearDoneRef.current)
        clearDoneRef.current = null
      }
    }
  }, [progress?.status])

  const runAnalyze = useCallback(async () => {
    setAnalyzeNote(null)
    try {
      await lichessApi.analyze()
    } catch (e: unknown) {
      const err = e as { response?: { status?: number; data?: { error?: string } } }
      if (err.response?.status === 409) {
        setAnalyzeNote(err.response.data?.error ?? 'Analysis already running.')
        return
      }
      setAnalyzeNote('Could not start analysis.')
      return
    }

    if (analyzePollRef.current != null) window.clearInterval(analyzePollRef.current)

    setAnalyzing(true)
    let sawRunning = false
    let polls = 0
    const maxPolls = 2400

    analyzePollRef.current = window.setInterval(async () => {
      polls += 1
      try {
        const { data } = await lichessApi.getAnalyzeStatus()
        if (data.running) sawRunning = true
        const finished =
          (sawRunning && !data.running) ||
          (!sawRunning && !data.running && polls >= 8)
        if (finished || polls >= maxPolls) {
          if (analyzePollRef.current != null) {
            window.clearInterval(analyzePollRef.current)
            analyzePollRef.current = null
          }
          setAnalyzing(false)
          queryClient.invalidateQueries({ queryKey: ['lichess'] })
        }
      } catch {
        if (analyzePollRef.current != null) {
          window.clearInterval(analyzePollRef.current)
          analyzePollRef.current = null
        }
        setAnalyzing(false)
      }
    }, 1500)
  }, [queryClient])

  const showSyncPill =
    streaming || progress?.status === 'done' || progress?.status === 'error'

  return (
    <div className="flex items-center gap-2 shrink-0">
      {showSyncPill && (
        <span
          className={
            streaming
              ? 'inline-flex items-center gap-1.5 rounded-full border border-amber-500/35 bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-medium text-amber-300'
              : progress?.status === 'error'
                ? 'inline-flex max-w-[10rem] truncate rounded-full border border-red-500/35 bg-red-500/10 px-2.5 py-0.5 text-[11px] font-medium text-red-300'
                : 'inline-flex items-center gap-1 rounded-full border border-emerald-500/35 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-300'
          }
          title={progress?.error ?? undefined}
        >
          {streaming && (
            <>
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400 animate-pulse" aria-hidden />
              Syncing
            </>
          )}
          {!streaming && progress?.status === 'done' && 'Synced'}
          {!streaming && progress?.status === 'error' && (progress.error ? `Failed: ${progress.error}` : 'Sync failed')}
        </span>
      )}

      {analyzing && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-violet-500/35 bg-violet-500/10 px-2.5 py-0.5 text-[11px] font-medium text-violet-200">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-violet-400 animate-pulse" aria-hidden />
          Analyzing
        </span>
      )}

      <button
        type="button"
        onClick={() => void runSync()}
        disabled={streaming}
        title="Fetch new games from Lichess"
        className="rounded-md border border-zinc-600 bg-zinc-800/80 px-2.5 py-1 text-xs font-medium text-zinc-200 hover:bg-zinc-700 disabled:opacity-45 disabled:pointer-events-none transition-colors"
      >
        {streaming ? '…' : 'Sync'}
      </button>
      <button
        type="button"
        onClick={() => void runAnalyze()}
        disabled={streaming || analyzing}
        title={analyzeNote ?? 'Stockfish pass on stored games'}
        className="rounded-md border border-violet-500/40 bg-violet-500/10 px-2.5 py-1 text-xs font-medium text-violet-200 hover:bg-violet-500/20 disabled:opacity-45 disabled:pointer-events-none transition-colors"
      >
        {analyzing ? '…' : 'Analyze'}
      </button>
    </div>
  )
}
