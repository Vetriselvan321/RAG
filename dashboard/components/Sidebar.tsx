'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  MessageSquare, Zap, ImageIcon, Upload, Settings,
  ChevronLeft, ChevronRight, Brain,
  Activity, BookOpen,
} from 'lucide-react'

// Hostname of the backend (strip protocol) for the status indicator.
const API_HOST = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000')
  .replace(/^https?:\/\//, '')

const NAV = [
  {
    href: '/',
    label: 'Ask',
    icon: MessageSquare,
    sub: 'Chat with your documents',
  },
  {
    href: '/stream',
    label: 'Stream',
    icon: Zap,
    sub: 'Real-time token streaming',
  },
  {
    href: '/ask-image',
    label: 'Ask Image',
    icon: ImageIcon,
    sub: 'Vision + RAG',
  },
  {
    href: '/upload',
    label: 'Upload',
    icon: Upload,
    sub: 'Add to knowledge base',
  },
  {
    href: '/settings',
    label: 'Settings',
    icon: Settings,
    sub: 'Manage API keys',
  },
]


const PILLS = [
  { label: 'Hybrid Search', color: 'bg-accent-blue/10 text-accent-blue' },
  { label: 'Cross-Encoder', color: 'bg-accent-purple/10 text-accent-purple' },
  { label: 'Gemini Vision', color: 'bg-accent-green/10 text-accent-green' },
]

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const pathname = usePathname()

  return (
    <aside
      className={`relative flex flex-col h-screen flex-shrink-0 bg-bg-panel border-r border-white/[0.06] transition-all duration-300 ease-in-out ${collapsed ? 'w-[68px]' : 'w-64'}`}
    >
      {/* ── Logo ─────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-4 h-14 border-b border-white/[0.06] flex-shrink-0">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center flex-shrink-0 shadow-glow-blue">
          <Brain className="w-4 h-4 text-white" />
        </div>
        {!collapsed && (
          <div className="flex flex-col min-w-0 leading-tight">
            <span className="text-sm font-bold text-ink-primary truncate">DocuRetrieve</span>
            <span className="text-[10px] text-ink-muted truncate">A Document RAG Engine</span>
          </div>
        )}
      </div>

      {/* ── Navigation ───────────────────────────────────────────────── */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {!collapsed && (
          <p className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-widest text-ink-muted">
            Navigation
          </p>
        )}
        {NAV.map(({ href, label, icon: Icon, sub }) => {
          const active = pathname === href
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group ${
                active
                  ? 'nav-active'
                  : 'text-ink-secondary hover:text-ink-primary hover:bg-white/[0.04]'
              } ${collapsed ? 'justify-center' : ''}`}
            >
              <Icon
                className={`w-[18px] h-[18px] flex-shrink-0 transition-colors ${
                  active ? 'text-accent-blue' : 'text-ink-muted group-hover:text-ink-primary'
                }`}
              />
              {!collapsed && (
                <div className="leading-tight">
                  <div className={active ? 'text-accent-blue' : ''}>{label}</div>
                  <div className="text-[11px] text-ink-muted mt-0.5 font-normal">{sub}</div>
                </div>
              )}
            </Link>
          )
        })}
      </nav>

      {/* ── Tech stack pills ─────────────────────────────────────────── */}
      {!collapsed && (
        <div className="px-4 py-4 border-t border-white/[0.06] space-y-3">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-ink-muted">
            Powered by
          </p>
          <div className="flex flex-wrap gap-1.5">
            {PILLS.map(({ label, color }) => (
              <span
                key={label}
                className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${color}`}
              >
                {label}
              </span>
            ))}
          </div>
          {/* Status */}
          <div className="flex items-center gap-2">
            <Activity className="w-3 h-3 text-accent-green" />
            <span className="text-[11px] text-ink-muted">
              <span className="text-accent-green font-medium">API</span> ready · {API_HOST}
            </span>
          </div>
        </div>
      )}

      {/* ── Collapse toggle ───────────────────────────────────────────── */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        className="absolute -right-3 top-[70px] z-20 w-6 h-6 rounded-full bg-bg-elevated border border-white/[0.10] flex items-center justify-center text-ink-muted hover:text-ink-primary hover:border-white/20 transition-all shadow-card"
      >
        {collapsed ? (
          <ChevronRight className="w-3 h-3" />
        ) : (
          <ChevronLeft className="w-3 h-3" />
        )}
      </button>
    </aside>
  )
}
