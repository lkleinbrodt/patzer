import { useState } from 'react'

export interface Column<T> {
  key: string
  label: string
  sortable?: boolean
  render?: (row: T) => React.ReactNode
  className?: string
}

interface Props<T extends Record<string, unknown>> {
  columns: Column<T>[]
  rows: T[]
  rowKey: (row: T) => string
  rowClassName?: (row: T) => string
}

export default function SortableTable<T extends Record<string, unknown>>({
  columns,
  rows,
  rowKey,
  rowClassName,
}: Props<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sorted = sortKey
    ? [...rows].sort((a, b) => {
        const av = a[sortKey]
        const bv = b[sortKey]
        if (av == null) return 1
        if (bv == null) return -1
        const cmp = av < bv ? -1 : av > bv ? 1 : 0
        return sortDir === 'asc' ? cmp : -cmp
      })
    : rows

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800">
            {columns.map(col => (
              <th
                key={col.key}
                className={`py-2 px-3 text-left text-xs font-medium text-zinc-400 uppercase tracking-wider ${col.sortable ? 'cursor-pointer select-none hover:text-zinc-200' : ''} ${col.className ?? ''}`}
                onClick={col.sortable ? () => handleSort(col.key) : undefined}
              >
                {col.label}
                {col.sortable && sortKey === col.key && (
                  <span className="ml-1 text-amber-400">{sortDir === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(row => (
            <tr
              key={rowKey(row)}
              className={`border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors ${rowClassName?.(row) ?? ''}`}
            >
              {columns.map(col => (
                <td key={col.key} className={`py-2 px-3 ${col.className ?? ''}`}>
                  {col.render ? col.render(row) : String(row[col.key] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && (
        <div className="py-8 text-center text-zinc-500 text-sm">No data</div>
      )}
    </div>
  )
}
