import { NavLink } from 'react-router-dom'
import { LayoutDashboard, ScanSearch, Clock, Activity, Info, Satellite } from 'lucide-react'

const sections = [
  {
    label: 'OVERVIEW',
    items: [{ label: 'Mission Control', path: '/', Icon: LayoutDashboard }],
  },
  {
    label: 'ANALYSIS',
    items: [{ label: 'New Analysis', path: '/analysis', Icon: ScanSearch }],
  },
  {
    label: 'INTELLIGENCE',
    items: [{ label: 'Analysis History', path: '/history', Icon: Clock }],
  },
  {
    label: 'SYSTEM',
    items: [
      { label: 'System Status', path: '/status', Icon: Activity },
      { label: 'About', path: '/about', Icon: Info },
    ],
  },
]

export default function Sidebar() {
  return (
    <div style={{
      width: 220,
      flexShrink: 0,
      background: 'var(--surface)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Logo */}
      <div style={{
        padding: '20px 20px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
      }}>
        <div style={{
          width: 32, height: 32,
          background: 'var(--accent)',
          borderRadius: 8,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <Satellite size={18} color="#fff" />
        </div>
        <div>
          <div style={{ fontSize: '0.85rem', fontWeight: 800, letterSpacing: '0.08em', lineHeight: 1.1 }}>SATQUERY</div>
          <div style={{ fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.18em', color: 'var(--accent)', lineHeight: 1.2 }}>AI PLATFORM</div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
        {sections.map(section => (
          <div key={section.label} style={{ marginBottom: 20 }}>
            <div className="sq-label" style={{ padding: '0 20px', marginBottom: 4 }}>{section.label}</div>
            {section.items.map(({ label, path, Icon }) => (
              <NavLink
                key={path}
                to={path}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
                end={path === '/'}
              >
                <Icon size={14} />
                {label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      {/* Version badge */}
      <div style={{
        padding: '12px 20px',
        borderTop: '1px solid var(--border)',
        fontSize: '0.65rem',
        color: 'var(--text-sub)',
        letterSpacing: '0.06em',
      }}>
        SatQuery AI v0.1.0
      </div>
    </div>
  )
}
