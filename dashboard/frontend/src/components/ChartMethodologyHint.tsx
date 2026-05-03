import { useEffect, useRef, useState, type ReactNode } from 'react'

export type ChartMethodologySection = {
  title: string
  /** Short paragraph under the title */
  detail: ReactNode
}

type ChartMethodologyHintProps = {
  sections: ChartMethodologySection[]
  /** Muted line at the bottom (stats, caveats, etc.) */
  footnote?: ReactNode
}

/** Explains how to read the chart — hover or click the control. */
export default function ChartMethodologyHint({ sections, footnote }: ChartMethodologyHintProps) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const cancelClose = () => {
    if (leaveTimer.current != null) {
      clearTimeout(leaveTimer.current)
      leaveTimer.current = undefined
    }
  }

  const scheduleClose = () => {
    cancelClose()
    leaveTimer.current = window.setTimeout(() => setOpen(false), 180)
  }

  useEffect(() => () => cancelClose(), [])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current != null && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  return (
    <div
      ref={wrapRef}
      className="relative ml-1 inline-flex align-middle"
      onPointerEnter={() => {
        cancelClose()
        setOpen(true)
      }}
      onPointerLeave={scheduleClose}
    >
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        aria-label="How to read this chart"
        className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-zinc-600 text-[10px] font-bold leading-none text-zinc-400 hover:border-zinc-500 hover:text-zinc-300"
      >
        ?
      </button>
      {open ? (
        <div
          role="tooltip"
          className="absolute left-0 top-full z-[100] mt-1 w-[min(22rem,calc(100vw-2rem))] rounded-md border border-zinc-600 bg-zinc-950 px-3 py-3 text-left shadow-xl"
          onPointerEnter={cancelClose}
          onPointerLeave={scheduleClose}
        >
          <div className="space-y-3">
            {sections.map((s, i) => (
              <div key={i}>
                <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">{s.title}</p>
                <div className="mt-1 text-xs leading-snug text-zinc-300 [&_strong]:font-medium [&_strong]:text-zinc-200">
                  {s.detail}
                </div>
              </div>
            ))}
          </div>
          {footnote != null && footnote !== '' ? (
            <p className="mt-3 border-t border-zinc-700/90 pt-2.5 text-[11px] leading-snug text-zinc-500">{footnote}</p>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
