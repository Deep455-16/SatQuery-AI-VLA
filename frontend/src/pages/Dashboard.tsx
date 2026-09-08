import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, Brain, Database, Activity, Zap, ChevronRight } from 'lucide-react'
import axios from 'axios'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'

const PROMPTS = [
  'Analyze vegetation density',
  'Detect water bodies',
  'Find built-up areas',
  'Describe the landscape',
  'Identify changes',
  'Detect flood damage',
]

export default function Dashboard() {
  const navigate = useNavigate()
  const [sysStatus, setSysStatus] = useState<Record<string, string>>({})
  const [historyCount, setHistoryCount] = useState<number | null>(null)
  const [queryInput, setQueryInput] = useState('')

  useEffect(() => {
    axios.get(`${API}/api/status`).then(r => setSysStatus(r.data)).catch(() => {})
    axios.get(`${API}/api/history?limit=999`).then(r => setHistoryCount(r.data.history?.length ?? 0)).catch(() => {})
  }, [])

  const handleQueryGo = () => {
    if (queryInput.trim()) navigate(`/analysis?q=${encodeURIComponent(queryInput.trim())}`)
    else navigate('/analysis')
  }

  const metrics = [
    {
      icon: Brain,
      label: 'AI ENGINE',
      value: sysStatus.ai_engine === 'operational' ? 'READY' : sysStatus.ai_engine ? 'OFFLINE' : '—',
      color: sysStatus.ai_engine === 'operational' ? 'var(--green)' : 'var(--text-muted)',
    },
    {
      icon: Database,
      label: 'DATABASE',
      value: sysStatus.mongodb === 'connected' ? 'CONNECTED' : sysStatus.mongodb ? 'OFFLINE' : '—',
      color: sysStatus.mongodb === 'connected' ? 'var(--green)' : 'var(--text-muted)',
    },
    {
      icon: Activity,
      label: 'ANALYSES RUN',
      value: historyCount !== null ? historyCount.toString() : '—',
      color: 'var(--text)',
    },
    {
      icon: Zap,
      label: 'API STATUS',
      value: sysStatus.api === 'connected' ? 'ONLINE' : sysStatus.api ? 'OFFLINE' : '—',
      color: sysStatus.api === 'connected' ? 'var(--green)' : 'var(--text-muted)',
    },
  ]

  return (
    <div className="fade-in" style={{ maxWidth: 1100, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* ── Hero ── */}
      <div className="sq-panel" style={{ position: 'relative', overflow: 'hidden', padding: '48px 56px' }}>
        {/* Grid bg */}
        <div className="hero-grid" style={{ position: 'absolute', inset: 0, borderRadius: 10 }} />
        {/* Accent gradient blob */}
        <div style={{
          position: 'absolute', top: -80, right: -60,
          width: 400, height: 400,
          background: 'radial-gradient(circle, var(--accent-dim) 0%, transparent 70%)',
          borderRadius: '50%',
          pointerEvents: 'none',
        }} />

        <div style={{ position: 'relative', zIndex: 1, maxWidth: 580 }}>
          <div className="chip accent" style={{ marginBottom: 20 }}>MISSION CONTROL</div>
          <h1 style={{ fontSize: '2.6rem', fontWeight: 900, lineHeight: 1.15, letterSpacing: '-0.02em', margin: '0 0 16px' }}>
            Remote-Sensing<br />Intelligence Console
          </h1>
          <p style={{ fontSize: '1rem', color: 'var(--text-muted)', lineHeight: 1.7, margin: '0 0 32px', maxWidth: 420 }}>
            Ask questions. Analyze imagery. Discover change.<br />
            Powered by multimodal Vision-Language AI.
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <button className="btn-primary" onClick={() => navigate('/analysis')}>
              START NEW ANALYSIS <ArrowRight size={14} />
            </button>
            <button className="btn-secondary" onClick={() => navigate('/history')}>
              VIEW HISTORY
            </button>
          </div>
        </div>
      </div>

      {/* ── Metrics ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14 }}>
        {metrics.map(({ icon: Icon, label, value, color }) => (
          <div key={label} className="sq-panel" style={{ padding: '18px 20px' }}>
            <Icon size={16} style={{ color: 'var(--text-muted)', marginBottom: 12 }} />
            <div style={{ fontSize: '1.35rem', fontWeight: 800, letterSpacing: '-0.01em', color, marginBottom: 4 }}>
              {value}
            </div>
            <div className="sq-label">{label}</div>
          </div>
        ))}
      </div>

      {/* ── Quick Query ── */}
      <div className="sq-panel" style={{ padding: '28px 32px' }}>
        <div className="sq-label" style={{ marginBottom: 16 }}>ASK SATQUERY</div>
        <div style={{ display: 'flex', gap: 10, marginBottom: 14 }}>
          <input
            className="sq-input"
            placeholder="What would you like to know about your imagery?"
            value={queryInput}
            onChange={e => setQueryInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleQueryGo()}
            style={{ flex: 1 }}
          />
          <button className="btn-primary" onClick={handleQueryGo} style={{ whiteSpace: 'nowrap' }}>
            ANALYZE <ArrowRight size={13} />
          </button>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {PROMPTS.map(p => (
            <button
              key={p}
              className="chip"
              style={{ cursor: 'pointer', transition: 'all 0.15s' }}
              onClick={() => navigate(`/analysis?q=${encodeURIComponent(p)}`)}
              onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = '')}
            >
              {p} <ChevronRight size={10} />
            </button>
          ))}
        </div>
      </div>

    </div>
  )
}
