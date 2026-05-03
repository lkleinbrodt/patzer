import { NavLink, useLocation } from 'react-router-dom'
import LichessHeaderActions from './LichessHeaderActions'

export default function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const lichessRoutes = location.pathname.startsWith('/lichess')

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 flex items-center gap-8 h-14 w-full min-w-0">
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-amber-400 font-bold text-lg tracking-tight">♟ Patzer</span>
            <span className="text-zinc-500 text-sm">dashboard</span>
          </div>
          <nav className="flex gap-1 min-w-0">
            <NavLink
              to="/eval"
              className={({ isActive }) =>
                `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-amber-500/15 text-amber-400'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`
              }
            >
              Local Evals
            </NavLink>
            <NavLink
              to="/lichess"
              className={({ isActive }) =>
                `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-amber-500/15 text-amber-400'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`
              }
            >
              Lichess
            </NavLink>
          </nav>
          {lichessRoutes && (
            <div className="ml-auto shrink-0 pl-2">
              <LichessHeaderActions />
            </div>
          )}
        </div>
      </header>
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">{children}</main>
    </div>
  )
}
