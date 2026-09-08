import { useState, useRef, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios'
import {
  Upload, X, Loader, ZoomIn, ZoomOut, Maximize2,
  Layers, Eye, CheckCircle, AlertCircle, Clock, Check, XCircle, FileJson, FileText, Trash2
} from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || ''

const MODES = ['Single Image', 'Before / After', 'Optical + SAR'] as const
type Mode = typeof MODES[number]

const PIPELINE_STEPS = [
  'Input validation',
  'Images detected & loaded',
  'Modality identification',
  'Task classification',
  'Compatibility verification',
  'Image preprocessing',
  'Evidence extraction',
  'Running TerraQ-VL',
  'Running specialist analysis',
  'Gemini synthesis',
  'Generating response',
]

interface TraceStep {
  step: string
  status: 'success' | 'failed' | 'skipped' | 'running'
  detail?: string
  duration_ms?: number
}

interface ExecutionTrace {
  task: string
  models: string[]
  tools: string[]
  parameters: Record<string, any>
  steps: TraceStep[]
  timing: Record<string, number>
  status: 'success' | 'partial' | 'failed'
  images_analyzed?: number
  images_failed?: number
}

interface AnalysisResult {
  task: string
  answer: string
  observations: string[]
  inferences: string[]
  evidence_strength: string
  limitations: string[]
  tools_used: string[]
  models_used: string[]
  processing_time_ms: number
  parameters: Record<string, any>
  warnings: string[]
  confidence: number | null
  execution_trace: ExecutionTrace | null
  images_failed?: number
  rgb_visualization?: string
  false_color_visualization?: string
  change_map?: string
  images_metadata?: any[]
  analysis_id?: string
  change_percentage?: number
}

const downloadJSON = (data: any, filename: string) => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

const downloadMD = (data: AnalysisResult, filename: string) => {
  let md = `# Analysis Report\n\n**Task:** ${data.task}\n\n## Assessment\n${data.answer}\n\n`
  if (data.observations?.length) md += `## Observations\n${data.observations.map(o => `- ${o}`).join('\n')}\n\n`
  if (data.inferences?.length) md += `## Inferences\n${data.inferences.map(i => `- ${i}`).join('\n')}\n\n`
  
  const blob = new Blob([md], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

const formatSize = (bytes: number) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const getModality = (filename: string) => {
  const lower = filename.toLowerCase()
  if (lower.includes('sar') || lower.includes('s1') || lower.includes('risat')) return 'SAR'
  return 'OPT'
}

export default function Analysis() {
  const [searchParams] = useSearchParams()
  const [mode, setMode] = useState<Mode>('Single Image')
  const [files, setFiles] = useState<File[]>([])
  const [dragging, setDragging] = useState(false)
  const [query, setQuery] = useState(searchParams.get('q') ?? '')
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState(-1)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState('')
  const [layer, setLayer] = useState<'RGB' | 'FALSE COLOR' | 'CHANGE MAP'>('RGB')
  const fileRef = useRef<HTMLInputElement>(null)
  const [fileUrls, setFileUrls] = useState<string[]>([])

  useEffect(() => {
    const urls = files.map(f => URL.createObjectURL(f))
    setFileUrls(urls)
    return () => urls.forEach(u => URL.revokeObjectURL(u))
  }, [files])

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    setFiles(prev => [...prev, ...Array.from(incoming)])
  }

  const runAnalysis = async () => {
    if (loading) return
    if (!files.length) return setError('Upload at least one satellite image.')
    if (!query.trim()) return setError('Provide a natural language query.')
    setError(''); setResult(null); setLoading(true); setStep(0)

    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    fd.append('query', query.trim())
    fd.append('mode', mode)

    const interval = setInterval(() => {
      setStep(s => (s < PIPELINE_STEPS.length - 2 ? s + 1 : s))
    }, 600)

    try {
      const res = await axios.post(`${API}/api/analyze`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      clearInterval(interval)
      setStep(PIPELINE_STEPS.length - 1)
      setResult(res.data)
      setLayer(res.data.change_map ? 'CHANGE MAP' : 'RGB')
    } catch (e: any) {
      clearInterval(interval)
      setError(e.response?.data?.error ?? 'Analysis failed. Check that the server is running.')
    } finally {
      setLoading(false)
      setTimeout(() => setStep(-1), 800)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === 'Enter') {
      runAnalysis()
    }
  }

  const currentImg = () => {
    if (!result) return null
    if (layer === 'FALSE COLOR' && result.false_color_visualization) return result.false_color_visualization
    if (layer === 'CHANGE MAP' && result.change_map) return result.change_map
    return result.rgb_visualization ?? null
  }

  const evidenceColor = (s: string) =>
    s === 'high' ? 'var(--green)' : s === 'moderate' ? 'var(--yellow)' : 'var(--red)'
  const evidencePct = (s: string) =>
    s === 'high' ? '100%' : s === 'moderate' ? '62%' : '28%'

  const suggestedQueries = {
    'Single Image': ['Describe the land-cover', 'Is there water visible?', 'Are there agricultural fields?', 'Describe vegetation', 'What infrastructure is visible?'],
    'Before / After': ['What changed between these dates?', 'Has built-up area increased?', 'Describe vegetation changes', 'What remained unchanged?'],
    'Optical + SAR': ['Use both images to identify built-up regions', 'What does SAR add to the optical analysis?', 'Identify water bodies using both modalities']
  }[mode]

  return (
    <div className="fade-in" style={{
      height: 'calc(100vh - 100px)',
      display: 'grid',
      gridTemplateColumns: '320px 1fr 340px',
      gap: 14,
    }}>

      {/* ══════════ LEFT: INPUT ══════════ */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto' }}>

        {/* Mode selector */}
        <div className="sq-panel" style={{ padding: 16 }}>
          <div className="sq-label" style={{ marginBottom: 10 }}>ANALYSIS MODE</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {MODES.map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  textAlign: 'left',
                  padding: '8px 12px',
                  borderRadius: 6,
                  border: `1px solid ${mode === m ? 'var(--accent)' : 'var(--border)'}`,
                  background: mode === m ? 'var(--accent-dim)' : 'transparent',
                  color: mode === m ? 'var(--accent)' : 'var(--text-muted)',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >{m}</button>
            ))}
          </div>
        </div>

        {/* File upload */}
        <div className="sq-panel" style={{ padding: 16, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div className="sq-label" style={{ margin: 0 }}>INPUT IMAGERY</div>
            {files.length > 0 && (
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {files.length} {files.length === 1 ? 'file' : 'files'} selected
              </div>
            )}
          </div>

          <div
            className={`drop-zone${dragging ? ' active' : ''}`}
            style={{ padding: '24px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', marginBottom: 10 }}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={20} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
            <div style={{ fontSize: '0.78rem', fontWeight: 600, marginBottom: 4 }}>DROP SATELLITE IMAGERY</div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-sub)' }}>TIFF · TIF · PNG · JPG</div>
            <input ref={fileRef} type="file" multiple style={{ display: 'none' }} onChange={e => addFiles(e.target.files)} />
          </div>

          {files.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 180, overflowY: 'auto', paddingRight: 4 }}>
              {files.map((f, i) => {
                const modality = getModality(f.name)
                return (
                  <div key={i} className="sq-panel-inner" style={{ display: 'flex', alignItems: 'center', padding: '6px', gap: 8 }}>
                    <div style={{ width: 32, height: 32, borderRadius: 4, overflow: 'hidden', background: '#000', flexShrink: 0 }}>
                      <img src={fileUrls[i]} alt="thumb" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    </div>
                    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
                        <span style={{ fontSize: '0.6rem', padding: '1px 4px', borderRadius: 3, background: modality === 'SAR' ? 'rgba(168,85,247,0.15)' : 'rgba(56,189,248,0.15)', color: modality === 'SAR' ? '#d8b4fe' : '#7dd3fc', fontWeight: 600 }}>
                          {modality}
                        </span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-sub)' }}>{formatSize(f.size)}</span>
                      </div>
                    </div>
                    <button onClick={() => setFiles(files.filter((_, j) => j !== i))} style={{ border: 'none', background: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 4, flexShrink: 0 }}>
                      <X size={14} />
                    </button>
                  </div>
                )
              })}
              <button 
                onClick={() => setFiles([])} 
                className="btn-ghost" 
                style={{ marginTop: 4, padding: '4px', fontSize: '0.7rem', color: 'var(--red)', justifyContent: 'center' }}
              >
                <Trash2 size={12} style={{ marginRight: 4 }} /> CLEAR ALL
              </button>
            </div>
          )}
        </div>

        {/* Query */}
        <div className="sq-panel" style={{ padding: 16 }}>
          <div className="sq-label" style={{ marginBottom: 10 }}>NATURAL LANGUAGE QUERY</div>
          <textarea
            className="sq-textarea"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g., Analyze vegetation density and identify areas showing signs of stress."
            style={{ minHeight: 88, resize: 'vertical', marginBottom: 10 }}
          />
          
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 12 }}>
            {suggestedQueries.map((sq, i) => (
              <button
                key={i}
                onClick={() => setQuery(sq)}
                style={{
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-muted)',
                  fontSize: '0.65rem',
                  padding: '4px 8px',
                  borderRadius: 12,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  transition: 'background 0.2s',
                }}
                onMouseOver={e => e.currentTarget.style.background = 'var(--border)'}
                onMouseOut={e => e.currentTarget.style.background = 'var(--surface-2)'}
              >
                {sq}
              </button>
            ))}
          </div>

          <div style={{ fontSize: '0.65rem', color: 'var(--text-sub)', marginBottom: 10, textAlign: 'right' }}>
            Ctrl+Enter to analyze
          </div>
          <button
            className="btn-primary"
            onClick={runAnalysis}
            disabled={loading}
            style={{ width: '100%', justifyContent: 'center', opacity: loading ? 0.7 : 1 }}
          >
            {loading ? <Loader size={13} className="spin" /> : null}
            {loading ? 'ANALYZING...' : 'ANALYZE'}
          </button>
          {error && (
            <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(220,38,38,0.08)', border: '1px solid rgba(220,38,38,0.2)', borderRadius: 6, fontSize: '0.75rem', color: 'var(--red)' }}>
              {error}
            </div>
          )}
        </div>
      </div>

      {/* ══════════ CENTER: VIEWER ══════════ */}
      <div className="sq-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Toolbar */}
        <div style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--surface-2)',
        }}>
          <div style={{ display: 'flex', gap: 4 }}>
            {[ZoomIn, ZoomOut, Maximize2].map((Icon, i) => (
              <button key={i} className="btn-ghost" style={{ padding: '5px 8px' }}>
                <Icon size={13} />
              </button>
            ))}
          </div>

          {result && (!result.images_metadata || result.images_metadata.length <= 1) && (
            <div style={{ display: 'flex', gap: 6 }}>
              {(['RGB', 'FALSE COLOR', 'CHANGE MAP'] as const).map(l => {
                const available = l === 'RGB' ? !!result.rgb_visualization :
                  l === 'FALSE COLOR' ? !!result.false_color_visualization :
                  !!result.change_map
                if (!available) return null
                return (
                  <button
                    key={l}
                    className={`btn-ghost${layer === l ? ' active' : ''}`}
                    onClick={() => setLayer(l)}
                    style={{ fontSize: '0.65rem', letterSpacing: '0.08em' }}
                  >
                    <Layers size={11} /> {l}
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Image area */}
        <div className="viewer-bg" style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', overflowY: 'auto', overflowX: 'hidden' }}>
          {loading ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <Loader size={36} className="spin" style={{ color: 'var(--accent)', marginBottom: 16 }} />
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
                PROCESSING MULTIMODAL PIPELINE...
              </div>
            </div>
          ) : result && (result.images_metadata && result.images_metadata.length > 1 || files.length > 1) ? (
            <div style={{ display: 'grid', gridTemplateColumns: (files.length <= 4 && files.length > 1) ? '1fr 1fr' : '1fr', gap: 12, padding: 16, height: '100%' }}>
              {(fileUrls.length > 0 ? fileUrls : []).map((url, i) => (
                <div key={i} style={{ position: 'relative', border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img src={url} alt={`Image ${i+1}`} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                  <div style={{ position: 'absolute', top: 8, left: 8, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', color: '#fff', fontSize: '0.7rem', padding: '2px 8px', borderRadius: 4, fontWeight: 600 }}>
                    Image {i + 1}
                  </div>
                </div>
              ))}
            </div>
          ) : currentImg() ? (
             <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
               <img src={currentImg()!} alt="Analysis result" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
             </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-sub)' }}>
              <Eye size={40} style={{ marginBottom: 12, opacity: 0.3 }} />
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', letterSpacing: '0.1em' }}>
                {files.length ? 'IMAGERY LOADED — AWAITING ANALYSIS' : 'NO IMAGERY LOADED'}
              </div>
            </div>
          )}
        </div>
        
        {/* Download actions */}
        {result && (
          <div style={{ padding: '8px 16px', borderTop: '1px solid var(--border)', display: 'flex', gap: 12, background: 'var(--surface)' }}>
            <button className="btn-ghost" onClick={() => downloadJSON(result, 'analysis_report.json')} style={{ fontSize: '0.7rem' }}>
              <FileJson size={14} style={{ marginRight: 6 }} /> Download JSON Report
            </button>
            <button className="btn-ghost" onClick={() => downloadMD(result, 'analysis_report.md')} style={{ fontSize: '0.7rem' }}>
              <FileText size={14} style={{ marginRight: 6 }} /> Download MD Report
            </button>
          </div>
        )}

        {/* Metadata strip */}
        {result?.images_metadata?.[0] && (
          <div style={{
            padding: '10px 16px',
            borderTop: '1px solid var(--border)',
            background: 'var(--surface-2)',
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 12,
          }}>
            {[
              ['SENSOR', result.images_metadata[0].sensor ?? 'Unknown'],
              ['RESOLUTION', result.images_metadata[0].resolution ?? '—'],
              ['BANDS', result.images_metadata[0].bands ?? '—'],
              ['CRS', result.images_metadata[0].crs ?? 'Unknown'],
            ].map(([k, v]) => (
              <div key={k}>
                <div className="sq-label">{k}</div>
                <div className="sq-value">{v}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ══════════ RIGHT: INTELLIGENCE ══════════ */}
      <div className="sq-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{
          padding: '10px 16px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--surface-2)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <Eye size={13} style={{ color: 'var(--text-muted)' }} />
          <div className="sq-label" style={{ margin: 0 }}>INTELLIGENCE</div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>

          {/* Pipeline steps while loading */}
          {loading && step >= 0 && (
            <div style={{ marginBottom: 20 }}>
              <div className="sq-label" style={{ marginBottom: 10 }}>SATQUERY PIPELINE</div>
              {PIPELINE_STEPS.map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-sub)', width: 20 }}>
                    {String(i+1).padStart(2,'0')}
                  </span>
                  {i < step ? <CheckCircle size={13} style={{ color: 'var(--green)', flexShrink: 0 }} /> :
                   i === step ? <Loader size={13} className="spin" style={{ color: 'var(--accent)', flexShrink: 0 }} /> :
                   <Clock size={13} style={{ color: 'var(--text-sub)', flexShrink: 0 }} />}
                  <span style={{
                    fontSize: '0.75rem',
                    color: i < step ? 'var(--text-muted)' : i === step ? 'var(--text)' : 'var(--text-sub)',
                    fontWeight: i === step ? 600 : 400,
                  }}>{s}</span>
                </div>
              ))}
            </div>
          )}

          {!result && !loading && (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-sub)' }}>
              <Eye size={36} style={{ marginBottom: 12, opacity: 0.3 }} />
              <div style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em', textAlign: 'center' }}>
                AWAITING<br />ANALYSIS
              </div>
            </div>
          )}

          {result && (
            <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

              {/* Task Classification Badge */}
              {result.task && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: -10 }}>
                  <div style={{ fontSize: '0.65rem', fontWeight: 700, padding: '4px 8px', borderRadius: 4, background: 'var(--accent-dim)', color: 'var(--accent)', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>
                    {result.task.replace(/_/g, ' ').toUpperCase()}
                  </div>
                </div>
              )}

              {/* Partial Failure Notice */}
              {(result.images_failed && result.images_failed > 0) ? (
                <div style={{ padding: '8px 12px', background: 'rgba(234,179,8,0.1)', border: '1px solid rgba(234,179,8,0.2)', borderRadius: 6, display: 'flex', gap: 8, alignItems: 'center' }}>
                  <AlertCircle size={16} style={{ color: 'var(--yellow)', flexShrink: 0 }} />
                  <span style={{ fontSize: '0.75rem', color: 'var(--yellow)', fontWeight: 500 }}>
                    Warning: {result.images_failed} image(s) failed to process completely.
                  </span>
                </div>
              ) : null}

              {/* Confidence & Evidence */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div className="sq-label">CONFIDENCE</div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text)' }}>
                    {typeof result.confidence === 'number' ? `${(result.confidence * 100).toFixed(1)}%` : 'Not available'}
                  </div>
                </div>
                {typeof result.confidence === 'number' && (
                  <div className="evidence-bar" style={{ marginBottom: 16 }}>
                    <div className="evidence-bar-fill" style={{
                      width: `${result.confidence * 100}%`,
                      background: result.confidence > 0.8 ? 'var(--green)' : result.confidence > 0.5 ? 'var(--yellow)' : 'var(--red)',
                    }} />
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div className="sq-label">EVIDENCE STRENGTH</div>
                  <div className={`chip ${result.evidence_strength === 'high' ? 'green' : result.evidence_strength === 'moderate' ? 'yellow' : 'red'}`}>
                    {(result.evidence_strength ?? 'limited').toUpperCase()}
                  </div>
                </div>
                <div className="evidence-bar">
                  <div className="evidence-bar-fill" style={{
                    width: evidencePct(result.evidence_strength ?? 'limited'),
                    background: evidenceColor(result.evidence_strength ?? 'limited'),
                  }} />
                </div>
              </div>

              {/* Change Percentage */}
              {(result.change_percentage !== undefined || result.parameters?.change_percentage !== undefined) && (
                <div style={{ padding: '12px', background: 'var(--surface-2)', borderRadius: 8, border: '1px solid var(--border)', textAlign: 'center' }}>
                  <div className="sq-label" style={{ marginBottom: 4 }}>CHANGE DETECTED</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent)' }}>
                    {Number(result.change_percentage ?? result.parameters?.change_percentage).toFixed(2)}%
                  </div>
                </div>
              )}

              {/* Answer */}
              {result.answer && (
                <div>
                  <div className="sq-label" style={{ marginBottom: 8, paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>AI ASSESSMENT</div>
                  <p style={{ fontSize: '0.82rem', lineHeight: 1.65, color: 'var(--text)', whiteSpace: '' }}>{result.answer}</p>
                </div>
              )}

              {/* Observations */}
              {result.observations?.length > 0 && (
                <div>
                  <div className="sq-label" style={{ marginBottom: 8, paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>OBSERVATIONS</div>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {result.observations.map((obs: string, i: number) => (
                      <li key={i} style={{ display: 'flex', gap: 8, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        <span style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 2 }}>◆</span>
                        {obs}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Inferences */}
              {result.inferences?.length > 0 && (
                <div>
                  <div className="sq-label" style={{ marginBottom: 8, paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>INFERENCES</div>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {result.inferences.map((inf: string, i: number) => (
                      <li key={i} style={{ display: 'flex', gap: 8, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        <span style={{ color: 'var(--yellow)', flexShrink: 0, marginTop: 2 }}>◆</span>
                        {inf}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Limitations */}
              {result.limitations?.length > 0 && (
                <div>
                  <div className="sq-label" style={{ marginBottom: 8, paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>LIMITATIONS</div>
                  {result.limitations.map((lim: string, i: number) => (
                    <div key={i} style={{ display: 'flex', gap: 8, fontSize: '0.75rem', color: 'var(--text-sub)', marginBottom: 4 }}>
                      <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 2, color: 'var(--yellow)' }} />
                      {lim}
                    </div>
                  ))}
                </div>
              )}

              {/* Execution Trace */}
              <div style={{ paddingTop: 12, borderTop: '1px solid var(--border)' }}>
                <div className="sq-label" style={{ marginBottom: 10 }}>EXECUTION TRACE</div>
                
                {result.execution_trace?.steps?.length ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {result.execution_trace.steps.map((st, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {st.status === 'success' ? <Check size={14} style={{ color: 'var(--green)' }} /> : 
                         st.status === 'failed' ? <XCircle size={14} style={{ color: 'var(--red)' }} /> : 
                         <Loader size={14} className="spin" style={{ color: 'var(--accent)' }} />}
                        <span style={{ flex: 1 }}>{st.step}</span>
                        {st.duration_ms !== undefined && (
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-sub)' }}>{st.duration_ms}ms</span>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="sq-panel-inner" style={{ padding: '10px 12px' }}>
                    {[
                      ['TASK', result.task ?? '—'],
                      ['PROCESSING', `${result.processing_time_ms ?? 0} ms`],
                      ['MODELS', result.models_used?.join(', ') ?? '—'],
                      ['TOOLS', result.tools_used?.join(', ') ?? '—'],
                    ].map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, gap: 8 }}>
                        <span className="sq-label" style={{ margin: 0 }}>{k}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text)', textAlign: 'right', wordBreak: 'break-all' }}>{v}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

            </div>
          )}
        </div>
      </div>

    </div>
  )
}
