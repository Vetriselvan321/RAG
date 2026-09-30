import ImageQueryForm from '@/components/ImageQueryForm'
import { ImageIcon, Eye, Search, Layers } from 'lucide-react'

const HOW = [
  { icon: Eye,    title: 'Image sent to Gemini 2.0 Flash Vision', sub: 'Full pixel-level analysis' },
  { icon: Search, title: 'Text query retrieves chunks from Qdrant', sub: 'Hybrid BM25 + vector search' },
  { icon: Layers, title: 'Gemini answers using image + context', sub: 'Grounded, cited response' },
]

export default function AskImagePage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto px-4 py-10">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-purple to-accent-blue flex items-center justify-center shadow-glow-purple">
              <ImageIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-ink-primary">Ask about an image</h1>
              <p className="text-sm text-ink-muted">Vision + RAG · Gemini 2.0 Flash</p>
            </div>
          </div>
          <p className="text-ink-secondary text-sm leading-relaxed">
            Upload a chart, diagram, screenshot, or photo. Gemini Vision analyzes it and
            cross-references your knowledge base to give a grounded, cited answer.
          </p>
        </div>

        {/* How it works */}
        <div className="glass rounded-2xl p-5 mb-6">
          <p className="text-xs font-semibold uppercase tracking-widest text-ink-muted mb-4">
            How multi-modal RAG works
          </p>
          <div className="space-y-3">
            {HOW.map(({ icon: Icon, title, sub }, i) => (
              <div key={title} className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-accent-purple/10 border border-accent-purple/20 flex items-center justify-center flex-shrink-0">
                  <Icon className="w-4 h-4 text-accent-purple" />
                </div>
                <div>
                  <p className="text-sm font-medium text-ink-primary">{title}</p>
                  <p className="text-xs text-ink-muted">{sub}</p>
                </div>
                {i < HOW.length - 1 && (
                  <div className="ml-4 w-px h-4 bg-white/[0.06] absolute" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Form */}
        <div className="glass rounded-2xl p-6">
          <ImageQueryForm />
        </div>

        <p className="text-center text-xs text-ink-muted mt-4">
          You can also{' '}
          <a href="/upload" className="text-accent-purple hover:underline">
            upload images as documents
          </a>{' '}
          — Gemini describes them so they become text-searchable.
        </p>
      </div>
    </div>
  )
}
