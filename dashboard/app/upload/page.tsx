import UploadForm from '@/components/UploadForm'
import { Database, Cpu, Layers } from 'lucide-react'

const STEPS = [
  {
    icon: Database,
    title: 'Images described by Gemini Vision',
    sub: 'Other files are read directly',
  },
  {
    icon: Layers,
    title: 'Split into overlapping chunks',
    sub: '~512 tokens each, with context overlap',
  },
  {
    icon: Cpu,
    title: 'Embedded with all-MiniLM-L6-v2',
    sub: 'Runs locally — no API key needed',
  },
  {
    icon: Database,
    title: 'Stored in Qdrant',
    sub: 'Searchable via BM25 + vector retrieval',
  },
]

export default function UploadPage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto px-4 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-ink-primary mb-2">Upload documents</h1>
          <p className="text-ink-secondary text-sm">
            Add files to your knowledge base. Supported: PDF, TXT, Markdown, HTML, and images.
          </p>
        </div>

        {/* Form card */}
        <div className="glass rounded-2xl p-6 mb-6">
          <UploadForm />
        </div>

        {/* Pipeline steps */}
        <div className="glass rounded-2xl p-6">
          <p className="text-xs font-semibold uppercase tracking-widest text-ink-muted mb-4">
            What happens after upload
          </p>
          <ol className="space-y-4">
            {STEPS.map(({ icon: Icon, title, sub }, i) => (
              <li key={title} className="flex items-start gap-3">
                <div className="flex-shrink-0 w-7 h-7 rounded-lg bg-accent-blue/10 border border-accent-blue/20 flex items-center justify-center">
                  <span className="text-xs font-bold text-accent-blue">{i + 1}</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-ink-primary">{title}</p>
                  <p className="text-xs text-ink-muted mt-0.5">{sub}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  )
}
