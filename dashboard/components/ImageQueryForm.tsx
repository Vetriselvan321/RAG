'use client'

import { useRef, useState } from 'react'
import { askImage } from '@/lib/api'
import { useAPIKeys } from '@/context/APIKeyContext'
import {
  ImageIcon, X, Send, Loader2, AlertCircle,
  CheckCircle2, FileQuestion,
} from 'lucide-react'

const ACCEPTED = '.png,.jpg,.jpeg,.gif,.webp'

interface Citation { source: string; text: string }
interface Result {
  answer: string
  cited_sources: string[]
  citations: Citation[]
}

export default function ImageQueryForm() {
  const { geminiApiKey } = useAPIKeys()
  const inputRef = useRef<HTMLInputElement>(null)
  const [file,     setFile]     = useState<File | null>(null)
  const [preview,  setPreview]  = useState<string | null>(null)
  const [query,    setQuery]    = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState<Result | null>(null)
  const [error,    setError]    = useState<string | null>(null)

  function pick(f: File) {
    setFile(f)
    setResult(null)
    setError(null)
    setPreview(URL.createObjectURL(f))
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) pick(f)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await askImage(
      file,
      query || undefined,
      5,
      geminiApiKey  
      )
      setResult({
        answer: data.answer,
        cited_sources: data.cited_sources,
        citations: data.citations as Citation[],
      })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* ── Image drop zone ─────────────────────────────────────────── */}
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        className={`drop-zone cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-200 overflow-hidden ${
          dragOver
            ? 'drag-over border-accent-purple/60 bg-accent-purple/5'
            : preview
              ? 'border-accent-purple/30'
              : 'border-white/10 hover:border-accent-purple/40'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) pick(f) }}
        />

        {preview ? (
          <div className="relative group">
            <img
              src={preview}
              alt="Preview"
              className="w-full max-h-64 object-contain p-4"
            />
            <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-2xl">
              <p className="text-white text-sm font-medium">Click to change image</p>
            </div>
            <button
              type="button"
              onClick={e => {
                e.stopPropagation()
                setFile(null)
                setPreview(null)
                setResult(null)
              }}
              className="absolute top-3 right-3 w-7 h-7 rounded-full bg-black/60 backdrop-blur-sm border border-white/20 flex items-center justify-center text-white hover:bg-black/80 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 p-12 text-center">
            <div className="w-16 h-16 rounded-2xl bg-bg-elevated border border-white/[0.06] flex items-center justify-center">
              <ImageIcon className="w-8 h-8 text-ink-muted" />
            </div>
            <div>
              <p className="font-semibold text-ink-primary">
                Drop an image or{' '}
                <span className="text-accent-purple">browse</span>
              </p>
              <p className="text-sm text-ink-muted mt-1">
                PNG · JPEG · GIF · WebP
              </p>
            </div>
          </div>
        )}
      </div>

      {file && (
        <p className="text-xs text-ink-muted -mt-2">
          <span className="font-mono text-ink-secondary">{file.name}</span>
          {' · '}{(file.size / 1024).toFixed(1)} KB
        </p>
      )}

      {/* ── Query input ─────────────────────────────────────────────── */}
      <div className="relative input-glow rounded-xl bg-bg-card border border-white/[0.08] transition-all">
        <FileQuestion className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-muted pointer-events-none" />
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="What does this chart show? (optional — defaults to full description)"
          disabled={loading}
          className="w-full bg-transparent text-ink-primary placeholder-ink-muted py-3.5 pl-11 pr-4 text-sm outline-none"
        />
      </div>

      {/* ── Submit ──────────────────────────────────────────────────── */}
      <button
        type="submit"
        disabled={!file || loading}
        className="w-full flex items-center justify-center gap-2.5 py-3.5 rounded-xl bg-gradient-to-r from-accent-purple to-accent-purple/80 text-white font-semibold text-sm disabled:opacity-30 disabled:cursor-not-allowed hover:shadow-glow-purple transition-all duration-200"
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyzing with Gemini Vision…
          </>
        ) : (
          <>
            <Send className="w-4 h-4" />
            Analyze Image
          </>
        )}
      </button>

      {/* ── Error ───────────────────────────────────────────────────── */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-accent-red/10 border border-accent-red/20 text-accent-red text-sm animate-fade-in">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {/* ── Result ──────────────────────────────────────────────────── */}
      {result && (
        <div className="animate-fade-in space-y-3">
          <div className="p-4 rounded-xl bg-bg-card border border-white/[0.06]">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="w-4 h-4 text-accent-green" />
              <span className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
                Gemini Vision Answer
              </span>
            </div>
            <p className="text-sm text-ink-primary leading-relaxed whitespace-pre-wrap">
              {result.answer}
            </p>
          </div>

          {result.cited_sources.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {result.cited_sources.map((src, j) => (
                <span
                  key={src}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-accent-purple/10 border border-accent-purple/20 text-[11px] font-medium text-accent-purple"
                >
                  <span className="w-4 h-4 rounded-full bg-accent-purple text-white text-[9px] flex items-center justify-center font-bold">
                    {j + 1}
                  </span>
                  <span className="font-mono truncate max-w-[140px]">{src}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </form>
  )
}
