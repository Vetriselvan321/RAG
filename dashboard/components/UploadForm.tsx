'use client'

import { useCallback, useRef, useState } from 'react'
import { uploadDocument } from '@/lib/api'
import {
  CloudUpload, FileText, FileCode, File, X,
  CheckCircle2, AlertCircle, Loader2,
} from 'lucide-react'

const ACCEPTED_EXTS  = ['.pdf', '.txt', '.md', '.html', '.htm', '.png', '.jpg', '.jpeg', '.gif', '.webp']
const ACCEPTED_ATTR  = ACCEPTED_EXTS.join(',')

function fileIcon(name: string) {
  const ext = name.split('.').pop()?.toLowerCase()
  if (ext === 'pdf') return FileText
  if (['md', 'txt'].includes(ext ?? '')) return FileCode
  return File
}

function formatBytes(n: number) {
  if (n < 1024)        return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}

export default function UploadForm() {
  const [file,    setFile]    = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [success, setSuccess] = useState<{ filename: string; chunks: number } | null>(null)
  const [error,   setError]   = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const pick = useCallback((f: File) => {
    setFile(f)
    setSuccess(null)
    setError(null)
    setProgress(0)
  }, [])

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
    setSuccess(null)
    setProgress(0)

    // Simulate incremental progress (real upload doesn't have progress events here)
    const interval = setInterval(() => {
      setProgress(p => Math.min(p + Math.random() * 15, 85))
    }, 250)

    try {
      const data = await uploadDocument(file)
      clearInterval(interval)
      setProgress(100)
      setSuccess({ filename: data.filename, chunks: data.chunks_stored })
      setFile(null)
      if (inputRef.current) inputRef.current.value = ''
    } catch (err: unknown) {
      clearInterval(interval)
      setProgress(0)
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const FileIcon = file ? fileIcon(file.name) : CloudUpload

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* ── Drop zone ───────────────────────────────────────────────── */}
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        className={`drop-zone cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-200 ${
          dragOver
            ? 'drag-over border-accent-blue/60 bg-accent-blue/5'
            : file
              ? 'border-accent-blue/40 bg-accent-blue/5'
              : 'border-white/10 hover:border-accent-blue/40'
        }`}
      >
        <input
          ref={inputRef}
          id="file-input"
          type="file"
          accept={ACCEPTED_ATTR}
          className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) pick(f) }}
        />

        {file ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-14 h-14 rounded-2xl bg-accent-blue/15 border border-accent-blue/20 flex items-center justify-center">
              <FileIcon className="w-7 h-7 text-accent-blue" />
            </div>
            <div>
              <p className="font-semibold text-ink-primary">{file.name}</p>
              <p className="text-sm text-ink-muted mt-0.5">{formatBytes(file.size)}</p>
            </div>
            <button
              type="button"
              onClick={e => { e.stopPropagation(); setFile(null) }}
              className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-accent-red transition-colors"
            >
              <X className="w-3 h-3" /> Remove
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-bg-elevated border border-white/[0.06] flex items-center justify-center">
              <CloudUpload className="w-8 h-8 text-ink-muted" />
            </div>
            <div>
              <p className="font-semibold text-ink-primary">
                Drop your file here, or{' '}
                <span className="text-accent-blue">browse</span>
              </p>
              <p className="text-sm text-ink-muted mt-1">
                PDF · TXT · Markdown · HTML · PNG · JPG · WebP
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-1.5 mt-1">
              {ACCEPTED_EXTS.map(ext => (
                <span
                  key={ext}
                  className="px-2 py-0.5 rounded-md bg-bg-card border border-white/[0.06] text-[10px] font-mono text-ink-muted"
                >
                  {ext}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Progress bar ────────────────────────────────────────────── */}
      {loading && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-ink-muted">
            <span>Uploading &amp; ingesting…</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-bg-elevated overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent-blue to-accent-purple rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* ── Submit button ────────────────────────────────────────────── */}
      <button
        type="submit"
        disabled={!file || loading}
        className="w-full flex items-center justify-center gap-2.5 py-3.5 rounded-xl bg-gradient-to-r from-accent-blue to-accent-blue/90 text-white font-semibold text-sm disabled:opacity-30 disabled:cursor-not-allowed hover:shadow-glow-blue transition-all duration-200"
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing…
          </>
        ) : (
          <>
            <CloudUpload className="w-4 h-4" />
            Upload &amp; Ingest
          </>
        )}
      </button>

      {/* ── Error ───────────────────────────────────────────────────── */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-accent-red/10 border border-accent-red/20 text-accent-red text-sm animate-fade-in">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-semibold">Upload failed</p>
            <p className="text-accent-red/80 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* ── Success ─────────────────────────────────────────────────── */}
      {success && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-accent-green/10 border border-accent-green/20 animate-fade-in">
          <CheckCircle2 className="w-5 h-5 text-accent-green mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-semibold text-accent-green">Successfully ingested</p>
            <p className="text-sm text-ink-secondary mt-0.5">
              <span className="font-mono text-ink-primary">{success.filename}</span>
              {' — '}{success.chunks} chunks stored in Qdrant
            </p>
          </div>
        </div>
      )}
    </form>
  )
}
