import { Satellite, Brain, Layers, GitCompare, ShieldCheck } from 'lucide-react'

const FEATURES = [
  {
    Icon: Layers,
    title: 'MULTIMODAL',
    sub: 'OPTICAL + SAR',
    desc: 'Joint reasoning over optical and synthetic-aperture radar imagery for comprehensive scene understanding.',
  },
  {
    Icon: GitCompare,
    title: 'TEMPORAL',
    sub: 'BEFORE / AFTER',
    desc: 'Bi-temporal change detection using OpenCV-based change mapping and Gemini evidence synthesis.',
  },
  {
    Icon: Brain,
    title: 'VISION-LANGUAGE',
    sub: 'NATURAL LANGUAGE',
    desc: 'Ask complex geospatial questions in plain English. The system classifies intent and routes to the correct analysis pipeline.',
  },
  {
    Icon: ShieldCheck,
    title: 'EVIDENCE',
    sub: 'AUDITABLE ANALYSIS',
    desc: 'Every inference is grounded in deterministic image measurements. Anti-hallucination prompting prevents fabricated outputs.',
  },
]

export default function About() {
  return (
    <div className="fade-in" style={{ maxWidth: 820, margin: '0 auto', paddingBottom: 48 }}>

      {/* Hero */}
      <div className="sq-panel" style={{ padding: '52px 56px', textAlign: 'center', marginBottom: 28, position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', inset: 0, borderRadius: 10,
          background: 'radial-gradient(ellipse at 50% 0%, var(--accent-dim) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{
            width: 60, height: 60,
            background: 'var(--accent)',
            borderRadius: 14,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 24px',
          }}>
            <Satellite size={28} color="#fff" />
          </div>
          <h1 style={{ fontSize: '2.4rem', fontWeight: 900, letterSpacing: '-0.02em', margin: '0 0 8px' }}>SATQUERY AI</h1>
          <div className="sq-label" style={{ letterSpacing: '0.2em', marginBottom: 20, display: 'block' }}>
            INTELLIGENCE FOR EARTH OBSERVATION
          </div>
          <p style={{ fontSize: '1rem', color: 'var(--text-muted)', maxWidth: 520, margin: '0 auto', lineHeight: 1.7 }}>
            SatQuery AI transforms natural-language questions into evidence-grounded analysis of multimodal remote-sensing imagery using agentic Vision-Language AI.
          </p>
        </div>
      </div>

      {/* Features */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 28 }}>
        {FEATURES.map(({ Icon, title, sub, desc }) => (
          <div key={title} className="sq-panel" style={{ padding: '22px 24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
              <div className="sq-panel-inner" style={{ padding: '8px 10px', display: 'inline-flex' }}>
                <Icon size={16} style={{ color: 'var(--accent)' }} />
              </div>
              <div>
                <div style={{ fontWeight: 800, fontSize: '0.88rem', letterSpacing: '0.06em' }}>{title}</div>
                <div className="sq-label" style={{ color: 'var(--accent)', margin: 0 }}>{sub}</div>
              </div>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>{desc}</p>
          </div>
        ))}
      </div>

      {/* Origin */}
      <div className="sq-panel" style={{ padding: '24px 32px', textAlign: 'center' }}>
        <div className="sq-label" style={{ marginBottom: 12 }}>PROJECT ORIGIN</div>
        <div style={{ fontWeight: 800, fontSize: '1.4rem', letterSpacing: '0.04em', marginBottom: 4 }}>SMART INDIA HACKATHON</div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 12 }}>ISRO / SAC — Problem Statement 26167</div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 10 }}>
          <span className="chip">Agentic AI</span>
          <span className="chip">Remote Sensing</span>
          <span className="chip">Vision-Language</span>
          <span className="chip">Earth Observation</span>
        </div>
      </div>

    </div>
  )
}
