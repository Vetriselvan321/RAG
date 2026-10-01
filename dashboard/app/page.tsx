'use client'

import { useEffect, useRef, useState } from 'react'

import { ask, SearchResult } from '@/lib/api'

import { useAPIKeys } from '@/context/APIKeyContext'

import {
  Send, Brain, Sparkles, BookOpen,
  FileText, Layers, FileQuestion, AlertCircle,
} from 'lucide-react'

interface Citation { source: string; text: string }

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  cited_sources?: SearchResult[]
  citations?: Citation[]
  error?: boolean
}
const SUGGESTIONS = [
  { icon: Brain,        text: 'What is Retrieval Augmented Generation?' },
  { icon: BookOpen,     text: 'Summarize the uploaded documents' },
  { icon: FileText,     text: 'What are the key topics covered?' },
  { icon: Layers,       text: 'Compare the different approaches mentioned' },
]

export default function AskPage() {
  const { groqApiKey } = useAPIKeys()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send(query?: string) {
    const q = (query ?? input).trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)

    const userMsg: Message = { id: `u-${Date.now()}`, role: 'user', content: q }
    setMessages(prev => [...prev, userMsg])

    try {
      const data = await ask(q, 5, groqApiKey)
      setMessages(prev => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: data.answer,
          cited_sources: data.cited_sources,
          citations: data.citations as Citation[],
        },
      ])
    } catch (err: unknown) {
      setMessages(prev => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content:
            err instanceof Error
              ? err.message
              : 'Something went wrong. Make sure the API server and Qdrant are running.',
          error: true,
        },
      ])
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  return (
    <div className="flex flex-col h-full">

      {/* ── Messages ─────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">

          {messages.length === 0 ? (
            /* ── Empty state ─────────────────────────────────────────── */
            <div className="flex flex-col items-center justify-center min-h-[calc(100vh-220px)] text-center animate-fade-in">
              {/* Icon */}
              <div className="relative mb-6">
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center shadow-glow-blue">
                  <Brain className="w-10 h-10 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-accent-green flex items-center justify-center">
                  <Sparkles className="w-2.5 h-2.5 text-white" />
                </div>
              </div>

              <h1 className="text-3xl font-bold text-ink-primary mb-3">
                What would you like to know?
              </h1>
              <p className="text-ink-secondary text-base max-w-md leading-relaxed mb-10">
                Ask anything about your uploaded documents. I use hybrid BM25 + vector search,
                cross-encoder reranking, and inline-cited LLM answers.
              </p>

              {/* Suggestion chips */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                {SUGGESTIONS.map(({ icon: Icon, text }) => (
                  <button
                    key={text}
                    onClick={() => send(text)}
                    className="flex items-start gap-3 px-4 py-3.5 rounded-xl bg-bg-card border border-white/[0.06] text-left text-sm text-ink-secondary hover:text-ink-primary hover:border-accent-blue/30 hover:bg-bg-elevated transition-all duration-200 group"
                  >
                    <Icon className="w-4 h-4 text-accent-blue mt-0.5 flex-shrink-0" />
                    <span className="leading-snug">{text}</span>
                  </button>
                ))}
              </div>

              <p className="mt-8 text-xs text-ink-muted">
                Upload documents via the{' '}
                <a href="/upload" className="text-accent-blue hover:underline">Upload</a>
                {' '}page first
              </p>
            </div>
          ) : (
            /* ── Chat messages ───────────────────────────────────────── */
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 animate-fade-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center flex-shrink-0 mt-0.5 shadow-glow-blue">
                    <Brain className="w-4 h-4 text-white" />
                  </div>
                )}

                <div className={`flex flex-col gap-2 ${msg.role === 'user' ? 'items-end' : 'items-start flex-1'}`}>
                  {/* Bubble */}
                  <div
                    className={`px-4 py-3 rounded-2xl text-sm leading-relaxed max-w-[85%] ${
                      msg.role === 'user'
                        ? 'bg-accent-blue text-white rounded-br-sm'
                        : msg.error
                          ? 'bg-accent-red/10 border border-accent-red/20 text-accent-red rounded-bl-sm w-full'
                          : 'bg-bg-card border border-white/[0.06] text-ink-primary rounded-bl-sm w-full'
                    }`}
                  >
                    {msg.error && (
                      <div className="flex items-center gap-2 mb-2">
                        <AlertCircle className="w-4 h-4 flex-shrink-0" />
                        <span className="font-semibold text-xs uppercase tracking-wide">Error</span>
                      </div>
                    )}
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>

                  {/* Citations */}
                  {msg.cited_sources && msg.cited_sources.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 max-w-[85%]">
                      {msg.cited_sources.map((src, j) => (
                        <span
                          key={src}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-accent-blue/10 border border-accent-blue/20 text-[11px] font-medium text-accent-blue"
                        >
                          <span className="w-4 h-4 rounded-full bg-accent-blue text-white text-[9px] flex items-center justify-center font-bold flex-shrink-0">
                            {j + 1}
                          </span>
                          <span className="font-mono truncate max-w-[140px]">{src}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Typing indicator */}
          {loading && (
            <div className="flex gap-3 animate-fade-in">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center flex-shrink-0 mt-0.5 shadow-glow-blue">
                <Brain className="w-4 h-4 text-white" />
              </div>
              <div className="bg-bg-card border border-white/[0.06] rounded-2xl rounded-bl-sm px-4 py-3.5">
                <div className="flex gap-1.5 items-center h-4">
                  {[0, 1, 2].map(i => (
                    <div
                      key={i}
                      className="typing-dot w-1.5 h-1.5 rounded-full bg-accent-blue"
                      style={{ animationDelay: `${i * 0.18}s` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Input bar ────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 border-t border-white/[0.06] bg-bg-primary/90 backdrop-blur-sm px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="relative input-glow rounded-2xl bg-bg-card border border-white/[0.08] transition-all duration-200">
            <div className="flex items-center">
              <FileQuestion className="absolute left-4 w-5 h-5 text-ink-muted pointer-events-none" />
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                placeholder="Ask anything about your uploaded documents…"
                disabled={loading}
                className="w-full bg-transparent text-ink-primary placeholder-ink-muted py-4 pl-12 pr-14 text-sm outline-none"
              />
              <button
                onClick={() => send()}
                disabled={!input.trim() || loading}
                className="absolute right-2 top-1/2 -translate-y-1/2 w-9 h-9 rounded-xl bg-accent-blue text-white flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed hover:bg-accent-blue/90 transition-all hover:shadow-glow-blue"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
          <p className="text-center text-[11px] text-ink-muted mt-2.5">
            Hybrid BM25 + Vector Search · Cross-Encoder Reranking · Inline Citations
          </p>
        </div>
      </div>
    </div>
  )
}
