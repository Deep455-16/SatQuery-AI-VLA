import { useEffect, useState } from 'react'
import axios from 'axios'
import { CheckCircle2, XCircle, Loader, Server, Brain, Database, RefreshCw } from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'

export default function Status() {
  const [status, setStatus] = useState<Record<string, string> | null>(null)
  const [checking, setChecking] = useState(true)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  const check = () => {
    setChecking(true)
    axios.get(`${API}/api/status`)
      .then(r => { setStatus(r.data); setLastChecked(new Date()) })
      .catch(() => setStatus({ ai_engine: 'offline', mongodb: 'disconnected', api: 'offline' }))
      .finally(() => setChecking(false))
  }

  useEffect(() => { check() }, [])

  const services = status ? [
    { icon: Server, title: 'API SERVER', state: status.api, desc: 'Flask REST API serving the frontend application.' },
    { icon: Brain, title: 'AI ENGINE', state: status.ai_engine, desc: 'Multimodal reasoning pipeline (Gemini Vision-Language).' },
    { icon: Database, title: 'DATABASE', state: status.mongodb, desc: 'MongoDB instance for history and metadata storage.' },
  ] : []

  const isOk = (s: string) => s === 'operational' || s === 'connected'

  return (
    <div className="fade-in" style={{ maxWidth: 900, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: '1.35rem', fontWeight: 800, letterSpacing: '-0.01em', margin: 0 }}>System Status</h1>
          {lastChecked && (
            <p style={{ margin: '4px 0 0', fontSize: '0.72rem', color: 'var(--text-sub)', fontFamily: 'var(--font-mono)' }}>
              Last checked: {lastChecked.toLocaleTimeString()}
            </p>
          )}
        </div>
        <button className="btn-secondary" onClick={check} disabled={checking}>
          {checking ? <Loader size={13} className="spin" /> : <RefreshCw size={13} />}
          {checking ? 'Checking...' : 'Refresh'}
        </button>
      </div>

      {checking && !status ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
          <Loader size={28} className="spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {services.map(({ icon: Icon, title, state, desc }) => (
            <div key={title} className="sq-panel" style={{ padding: '20px 22px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                <div className="sq-panel-inner" style={{ padding: '8px 10px', display: 'inline-flex' }}>
                  <Icon size={18} style={{ color: 'var(--text-muted)' }} />
                </div>
                {isOk(state)
                  ? <CheckCircle2 size={18} style={{ color: 'var(--green)' }} />
                  : <XCircle size={18} style={{ color: 'var(--red)' }} />}
              </div>
              <div className="sq-label" style={{ marginBottom: 4 }}>{title}</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 800, letterSpacing: '0.04em', marginBottom: 8, color: isOk(state) ? 'var(--green)' : 'var(--red)' }}>
                {(state ?? 'UNKNOWN').toUpperCase()}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{desc}</div>
            </div>
          ))}
        </div>
      )}

      {status?.version && (
        <div className="sq-panel" style={{ marginTop: 20, padding: '14px 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="sq-label">VERSION</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>{status.version}</div>
        </div>
      )}
    </div>
  )
}
