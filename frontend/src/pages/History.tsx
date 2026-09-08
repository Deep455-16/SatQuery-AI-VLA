import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  Clock, Search, Loader, ChevronRight, X,
  Eye, Lightbulb, AlertTriangle, Image, Bot, ChevronLeft
} from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'

const evidenceColor = (s: string) => {
  if (s === 'high') return { bg: 'rgba(34,197,94,0.12)', text: '#16a34a', border: '#86efac' }
  if (s === 'moderate') return { bg: 'rgba(234,179,8,0.12)', text: '#a16207', border: '#fde68a' }
  return { bg: 'rgba(100,116,139,0.12)', text: '#64748b', border: '#cbd5e1' }
}

const EvidenceBadge = ({ ev }: { ev: string }) => {
  const c = evidenceColor(ev)
  return (
    <span style={{
      background: c.bg, color: c.text, border: `1px solid ${c.border}`,
      borderRadius: 20, padding: '2px 10px', fontSize: '0.7rem', fontWeight: 700,
      textTransform: 'uppercase', letterSpacing: '0.05em'
    }}>
      {ev || 'unknown'}
    </span>
  )
}

export default function History() {
  const [history, setHistory] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<any | null>(null)

  useEffect(() => {
    axios.get(`${API}/api/history?limit=50`)
      .then(r => { setHistory(r.data.history ?? []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const filtered = history.filter(h =>
    !search ||
    h.query?.toLowerCase().includes(search.toLowerCase()) ||
    h.task?.toLowerCase().includes(search.toLowerCase()) ||
    h.answer?.toLowerCase().includes(search.toLowerCase())
  )

  const formatTime = (ts: number) => {
    if (!ts) return '—'
    const d = new Date(ts * 1000)
    const diff = (Date.now() - d.getTime()) / 1000
    if (diff < 60) return 'just now'
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
    return d.toLocaleDateString()
  }

  return (
    <div className="fade-in" style={{ display: 'flex', gap: 0, height: '100%', overflow: 'hidden' }}>

      {/* ── LIST COLUMN ── */}
      <div style={{
        width: selected ? 380 : '100%', minWidth: selected ? 340 : undefined,
        maxWidth: selected ? 420 : 900, margin: selected ? '0' : '0 auto',
        display: 'flex', flexDirection: 'column', transition: 'width 0.25s',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20, gap: 12, flexShrink: 0 }}>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.01em', margin: 0 }}>Analysis History</h1>
            <p style={{ margin: '3px 0 0', fontSize: '0.78rem', color: 'var(--text-muted)' }}>Intelligence reports from past sessions</p>
          </div>
          <div style={{ position: 'relative', flexShrink: 0 }}>
            <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-sub)' }} />
            <input
              className="sq-input"
              placeholder="Search analyses..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ paddingLeft: 30, width: 200 }}
            />
          </div>
        </div>

        {/* List */}
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
              <Loader size={28} className="spin" style={{ color: 'var(--accent)' }} />
            </div>
          ) : filtered.length === 0 ? (
            <div className="sq-panel" style={{ padding: 60, textAlign: 'center' }}>
              <Clock size={36} style={{ color: 'var(--text-sub)', marginBottom: 16, opacity: 0.4 }} />
              <div style={{ fontWeight: 700, marginBottom: 8 }}>{search ? 'No results found' : 'No history yet'}</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {search ? 'Try a different search term.' : 'Run analyses and they will appear here.'}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {filtered.map((item, i) => {
                const isActive = selected?.id === item.id
                return (
                  <div
                    key={i}
                    className="sq-panel"
                    onClick={() => setSelected(isActive ? null : item)}
                    style={{
                      padding: '14px 16px',
                      cursor: 'pointer',
                      borderColor: isActive ? 'var(--accent)' : undefined,
                      background: isActive ? 'rgba(59,130,246,0.04)' : undefined,
                      transition: 'border-color 0.15s, background 0.15s',
                    }}
                    onMouseEnter={e => { if (!isActive) e.currentTarget.style.borderColor = 'var(--accent)' }}
                    onMouseLeave={e => { if (!isActive) e.currentTarget.style.borderColor = '' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                      <span className="chip accent" style={{ fontSize: '0.62rem' }}>{item.task ?? 'vqa'}</span>
                      {item.evidence_strength && <EvidenceBadge ev={item.evidence_strength} />}
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-sub)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
                        {formatTime(item.timestamp)}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.83rem', fontWeight: 600, marginBottom: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      "{item.query}"
                    </div>
                    {item.answer && (
                      <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.answer}
                      </div>
                    )}
                    {/* Image filenames */}
                    {item.images_metadata?.length > 0 && (
                      <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                        {item.images_metadata.map((m: any, j: number) => (
                          <span key={j} style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: '0.63rem', color: 'var(--text-sub)', background: 'var(--surface-alt, #f1f5f9)', borderRadius: 4, padding: '1px 6px' }}>
                            <Image size={9} /> {m.filename || `Image ${j + 1}`}
                          </span>
                        ))}
                      </div>
                    )}
                    <ChevronRight size={14} style={{ position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-sub)', transition: 'transform 0.15s', ...(isActive ? { transform: 'translateY(-50%) rotate(90deg)' } : {}) }} />
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── DETAIL PANEL ── */}
      {selected && (
        <div style={{
          flex: 1, marginLeft: 16, overflowY: 'auto',
          borderLeft: '1px solid var(--border)',
          paddingLeft: 24,
          animation: 'slideInRight 0.2s ease',
        }}>
          {/* Panel header */}
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20, gap: 12, position: 'sticky', top: 0, background: 'var(--bg, #f8fafc)', paddingTop: 4, paddingBottom: 12, zIndex: 2 }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <span className="chip accent">{selected.task ?? 'vqa'}</span>
                {selected.evidence_strength && <EvidenceBadge ev={selected.evidence_strength} />}
                {selected.confidence != null && (
                  <span style={{ fontSize: '0.68rem', color: 'var(--text-sub)', fontFamily: 'var(--font-mono)' }}>
                    {Math.round(selected.confidence * 100)}% confidence
                  </span>
                )}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-sub)', fontFamily: 'var(--font-mono)' }}>
                {formatTime(selected.timestamp)}
                {selected.processing_time_ms ? ` · ${selected.processing_time_ms}ms` : ''}
                {selected.models_used?.length ? ` · ${selected.models_used[0]}` : ''}
              </div>
            </div>
            <button
              onClick={() => setSelected(null)}
              style={{ padding: 6, border: 'none', background: 'var(--border)', borderRadius: 6, cursor: 'pointer', color: 'var(--text-muted)', flexShrink: 0 }}
              title="Close"
            >
              <X size={14} />
            </button>
          </div>

          {/* Images used */}
          {selected.images_metadata?.length > 0 && (
            <div className="sq-panel" style={{ padding: '12px 16px', marginBottom: 16 }}>
              <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-sub)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
                📡 Input Imagery
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {selected.images_metadata.map((m: any, j: number) => (
                  <div key={j} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: '0.78rem' }}>
                    <Image size={13} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                    <div>
                      <div style={{ fontWeight: 600 }}>{m.filename || `Image ${j + 1}`}</div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                        {[m.modality, m.format, m.width && m.height && `${m.width}×${m.height}`, m.bands && `${m.bands} bands`, m.sensor].filter(Boolean).join(' · ')}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Question */}
          <div className="sq-panel" style={{ padding: '12px 16px', marginBottom: 16, borderLeft: '3px solid var(--accent)' }}>
            <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-sub)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
              💬 Query
            </div>
            <div style={{ fontSize: '0.88rem', fontWeight: 600, fontStyle: 'italic' }}>"{selected.query}"</div>
          </div>

          {/* AI Answer */}
          {selected.answer && (
            <div style={{ background: 'linear-gradient(135deg,#eff6ff,#f0f9ff)', borderLeft: '4px solid var(--accent)', borderRadius: 8, padding: '14px 16px', marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                <Bot size={13} style={{ color: 'var(--accent)' }} />
                <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>AI Assessment</span>
              </div>
              <div style={{ fontSize: '0.88rem', lineHeight: 1.7, color: 'var(--text-main)' }}>{selected.answer}</div>
            </div>
          )}

          {/* Observations */}
          {selected.observations?.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-main)' }}>
                <Eye size={13} style={{ color: '#22c55e' }} /> Observations
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {selected.observations.map((o: string, j: number) => (
                  <div key={j} style={{ background: 'white', border: '1px solid var(--border)', borderLeft: '3px solid #22c55e', borderRadius: 6, padding: '7px 12px', fontSize: '0.82rem', color: 'var(--text-main)' }}>
                    📍 {o}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Inferences */}
          {selected.inferences?.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-main)' }}>
                <Lightbulb size={13} style={{ color: '#f59e0b' }} /> Inferences
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {selected.inferences.map((inf: string, j: number) => (
                  <div key={j} style={{ background: 'white', border: '1px solid var(--border)', borderLeft: '3px solid #f59e0b', borderRadius: 6, padding: '7px 12px', fontSize: '0.82rem', color: 'var(--text-main)' }}>
                    → {inf}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Limitations */}
          {selected.limitations?.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-main)' }}>
                <AlertTriangle size={13} style={{ color: '#f59e0b' }} /> Limitations
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {selected.limitations.map((lim: string, j: number) => (
                  <div key={j} style={{ background: '#fefce8', border: '1px solid #fde68a', borderLeft: '3px solid #f59e0b', borderRadius: 6, padding: '7px 12px', fontSize: '0.8rem', color: '#78350f' }}>
                    ⚠️ {lim}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Warnings */}
          {selected.warnings?.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              {selected.warnings.map((w: string, j: number) => (
                <div key={j} style={{ fontSize: '0.75rem', color: 'var(--text-muted)', padding: '4px 0' }}>ℹ️ {w}</div>
              ))}
            </div>
          )}

          {/* Close / Back */}
          <button
            onClick={() => setSelected(null)}
            style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, padding: '8px 14px', border: '1px solid var(--border)', borderRadius: 8, background: 'none', cursor: 'pointer', fontSize: '0.78rem', color: 'var(--text-muted)' }}
          >
            <ChevronLeft size={13} /> Back to list
          </button>
        </div>
      )}

      <style>{`
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(20px); }
          to   { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  )
}
