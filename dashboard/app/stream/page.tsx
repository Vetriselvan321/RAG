'use client'

import { useEffect, useRef, useState } from 'react'
import { streamAsk } from '@/lib/api'
import { useAPIKeys } from '@/context/APIKeyContext'
import { Send, Zap, FileQuestion, AlertCircle, Square } from 'lucide-react'

interface Citation { source: string; text: string }
interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
  cited_sources?: string[]
  citations?: Citation[]
  error?: boolean
}

export default function StreamPage() {
  const { groqApiKey } = useAPIKeys()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput]       = useState('')
  const [streaming, setStreaming] = useState(false)
  const cancelRef = useRef(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(query?: string) {
    const q = (query ?? input).trim()
    if (!q || streaming) return
    setInput('')
    setStreaming(true)
    cancelRef.current = false

    const userId = `u-${Date.now()}`
    const aiId   = `a-${Date.now() + 1}`

    setMessages(prev => [
      ...prev,
      { id: userId, role: 'user', content: q },
      { id: aiId,   role: 'assistant', content: '', streaming: true },
    ])

    try {
      for await (const event of streamAsk(q, 5, groqApiKey)) {
        if (cancelRef.current) break

        if (event.type === 'token') {
          setMessages(prev =>
            prev.map(m =>
              m.id === aiId ? { ...m, content: m.content + event.text } : m,
            ),
          )
        } else if (event.type === 'citations') {
          setMessages(prev =>
            prev.map(m =>
              m.id === aiId
                ? {
                    ...m,
                    cited_sources: event.cited_sources,
                    citations: event.citations as Citation[],
                  }
                : m,
            ),
          )
        } else if (event.type === 'error') {
          setMessages(prev =>
            prev.map(m =>
              m.id === aiId ? { ...m, content: event.message, error: true } : m,
            ),
          )
        } else if (event.type === 'done') {
          break
        }
      }
    } catch (err: unknown) {
      setMessages(prev =>
        prev.map(m =>
          m.id === aiId
            ? {
                ...m,
                content: err instanceof Error ? err.message : 'Stream failed.',
                error: true,
              }
            : m,
        ),
      )
    } finally {
      setMessages(prev =>
        prev.map(m => (m.id === aiId ? { ...m, streaming: false } : m)),
      )
      setStreaming(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  function stop() {
    cancelRef.current = true
  }

  return (
    <div className="flex flex-col h-full">

      {/* ── Messages ─────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">

          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[calc(100vh-220px)] text-center animate-fade-in">
              <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center mb-6 shadow-glow-purple">
                <Zap className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-ink-primary mb-3">Streaming answers</h1>
              <p className="text-ink-secondary text-base max-w-md leading-relaxed mb-8">
                Same hybrid retrieval pipeline as Ask — but tokens appear instantly as the
                model generates them via Server-Sent Events.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {['How does RAG work?', 'What is cross-encoder reranking?', 'Explain hybrid search'].map(s => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="px-4 py-2 rounded-full bg-bg-card border border-white/[0.06] text-sm text-ink-secondary hover:text-ink-primary hover:border-accent-blue/30 transition-all"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map(msg => (
            <div
              key={msg.id}
              className={`flex gap-3 animate-fade-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center flex-shrink-0 mt-0.5 shadow-glow-purple">
                  <Zap className="w-4 h-4 text-white" />
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
                      <AlertCircle className="w-4 h-4" />
                      <span className="font-semibold text-xs uppercase tracking-wide">Error</span>
                    </div>
                  )}
                  <p className="whitespace-pre-wrap">
                    {msg.content || (msg.streaming ? '' : '…')}
                    {/* Blinking cursor while streaming */}
                    {msg.streaming && (
                      <span className="inline-block w-0.5 h-3.5 bg-accent-blue ml-0.5 align-middle animate-pulse" />
                    )}
                  </p>
                </div>

                {/* Streaming badge */}
                {msg.streaming && (
                  <div className="flex items-center gap-1.5 text-[11px] text-accent-blue">
                    <span className="relative flex h-1.5 w-1.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-blue opacity-75" />
                      <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-accent-blue" />
                    </span>
                    streaming…
                  </div>
                )}

                {/* Citations */}
                {!msg.streaming && msg.cited_sources && msg.cited_sources.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 max-w-[85%]">
                    {msg.cited_sources.map((src, j) => (
                      <span
                        key={src}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-accent-purple/10 border border-accent-purple/20 text-[11px] font-medium text-accent-purple"
                      >
                        <span className="w-4 h-4 rounded-full bg-accent-purple text-white text-[9px] flex items-center justify-center font-bold flex-shrink-0">
                          {j + 1}
                        </span>
                        <span className="font-mono truncate max-w-[140px]">{src}</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

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
                placeholder="Ask anything — answers stream token by token…"
                disabled={streaming}
                className="w-full bg-transparent text-ink-primary placeholder-ink-muted py-4 pl-12 pr-14 text-sm outline-none"
              />
              {streaming ? (
                <button
                  onClick={stop}
                  className="absolute right-2 top-1/2 -translate-y-1/2 w-9 h-9 rounded-xl bg-accent-red/20 text-accent-red border border-accent-red/30 flex items-center justify-center hover:bg-accent-red/30 transition-all"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                </button>
              ) : (
                <button
                  onClick={() => send()}
                  disabled={!input.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 w-9 h-9 rounded-xl bg-accent-purple text-white flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed hover:bg-accent-purple/90 transition-all hover:shadow-glow-purple"
                >
                  <Send className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
          <p className="text-center text-[11px] text-ink-muted mt-2.5">
            Server-Sent Events · Gemini streaming · Citations after generation
          </p>
        </div>
      </div>
    </div>
  )
}
