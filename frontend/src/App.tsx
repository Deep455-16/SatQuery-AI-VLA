import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import TopNav from './components/TopNav'
import Dashboard from './pages/Dashboard'
import Analysis from './pages/Analysis'
import History from './pages/History'
import Status from './pages/Status'
import About from './pages/About'

export default function App() {
  const [dark, setDark] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  return (
    <Router>
      <div style={{ display:'flex', height:'100vh', overflow:'hidden', background:'var(--bg)', color:'var(--text)' }}>
        <Sidebar />
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
          <TopNav dark={dark} onToggleDark={() => setDark(d => !d)} />
          <main style={{ flex:1, overflow:'auto', padding:'24px 28px' }}>
            <Routes>
              <Route path="/"         element={<Dashboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/history"  element={<History />} />
              <Route path="/status"   element={<Status />} />
              <Route path="/about"    element={<About />} />
              <Route path="*"         element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  )
}
