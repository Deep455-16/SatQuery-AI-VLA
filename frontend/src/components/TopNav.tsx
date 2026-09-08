import { useEffect, useState } from 'react'
import { Moon, Sun, Bell } from 'lucide-react'
import axios from 'axios'

const API = import.meta.env.VITE_API_BASE_URL || ''

const StatusChip = ({ label, status }: { label: string, status: 'ok' | 'warn' | 'error' | 'loading' }) => {
  const colors = {
    ok: { bg: 'rgba(34,197,94,0.15)', text: '#16a34a', dot: '#16a34a' },
    warn: { bg: 'rgba(234,179,8,0.15)', text: '#a16207', dot: '#ca8a04' },
    error: { bg: 'rgba(239,68,68,0.15)', text: '#dc2626', dot: '#dc2626' },
    loading: { bg: 'rgba(100,116,139,0.15)', text: '#64748b', dot: '#94a3b8' },
  }
  const color = colors[status]

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      background: color.bg,
      padding: '4px 10px',
      borderRadius: 12,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%', background: color.dot,
        boxShadow: status === 'ok' ? `0 0 4px ${color.dot}` : 'none'
      }} />
      <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: color.text, letterSpacing: '0.06em', fontWeight: 600 }}>
        {label}
      </span>
    </div>
  )
}

export default function TopNav({ dark, onToggleDark }: { dark: boolean; onToggleDark: () => void }) {
  const [healthStatus, setHealthStatus] = useState<'ok' | 'error' | 'loading'>('loading')
  const [healthText, setHealthText] = useState('Connected')
  const [terraqStatus, setTerraqStatus] = useState<'ok' | 'warn' | 'error' | 'loading'>('loading')
  const [terraqText, setTerraqText] = useState('READY')
  const [geminiStatus, setGeminiStatus] = useState<'ok' | 'error' | 'loading'>('loading')
  const [geminiText, setGeminiText] = useState('CONFIGURED')

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [healthRes, modelsRes] = await Promise.all([
          axios.get(`${API}/api/health`).catch(() => null),
          axios.get(`${API}/api/models/status`).catch(() => null)
        ])

        if (healthRes && healthRes.data) {
          setHealthStatus('ok')
          setHealthText('Connected')
        } else {
          setHealthStatus('error')
          setHealthText('Disconnected')
        }

        if (modelsRes && modelsRes.data) {
          const tStatus = modelsRes.data.terraq_vl?.status
          if (tStatus === 'READY') {
            setTerraqStatus('ok')
            setTerraqText('READY')
          } else if (tStatus === 'STANDBY') {
            setTerraqStatus('ok')           // Green — standby is a healthy state
            setTerraqText('STANDBY (Gemini)')
          } else if (tStatus === 'BLOCKED' || tStatus === 'UNAVAILABLE') {
            setTerraqStatus('ok')           // Still show green per UI requirement
            setTerraqText('STANDBY (Gemini)')
          } else if (tStatus === 'ERROR') {
            setTerraqStatus('error')
            setTerraqText('ERROR')
          } else {
            setTerraqStatus('ok')           // Default to green
            setTerraqText('STANDBY (Gemini)')
          }

          const gStatus = modelsRes.data.gemini?.status
          if (gStatus === 'CONFIGURED' || gStatus === 'CONNECTED') {
            setGeminiStatus('ok')
            setGeminiText('CONNECTED')
          } else {
            setGeminiStatus('error')
            setGeminiText(gStatus || 'MISSING KEY')
          }
        } else {
          setTerraqStatus('error')
          setTerraqText('ERROR')
          setGeminiStatus('error')
          setGeminiText('ERROR')
        }
      } catch (err) {
        setHealthStatus('error')
        setHealthText('Disconnected')
        setTerraqStatus('error')
        setTerraqText('ERROR')
        setGeminiStatus('error')
        setGeminiText('ERROR')
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <header style={{
      height: 52,
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div className="sq-label" style={{ letterSpacing: '0.16em', margin: 0 }}>
          REMOTE-SENSING INTELLIGENCE CONSOLE
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-sub)', fontFamily: 'var(--font-mono)' }}>
          SatQuery AI v1.0 · PS 26167
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        {/* Status indicators removed per user request */}

        {/* Divider */}
        <div style={{ width: 1, height: 20, background: 'var(--border)' }} />

        {/* Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            onClick={onToggleDark}
            style={{ padding: 6, border: 'none', background: 'none', color: 'var(--text-muted)', cursor: 'pointer', borderRadius: 6 }}
            title="Toggle theme"
          >
            {dark ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          <button style={{ padding: 6, border: 'none', background: 'none', color: 'var(--text-muted)', cursor: 'pointer', borderRadius: 6 }}>
            <Bell size={15} />
          </button>
        </div>
      </div>
    </header>
  )
}
