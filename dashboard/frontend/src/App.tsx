import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import EvalLeaderboard from './pages/EvalLeaderboard'
import LichessAnalysis from './pages/LichessAnalysis'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/eval" replace />} />
        <Route path="/eval" element={<EvalLeaderboard />} />
        <Route path="/lichess" element={<LichessAnalysis />} />
        <Route path="/lichess/performance" element={<Navigate to="/lichess" replace />} />
      </Routes>
    </Layout>
  )
}
