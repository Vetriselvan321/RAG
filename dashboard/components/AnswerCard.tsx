import type { AskResponse } from '@/lib/api'
import { CheckCircle2 } from 'lucide-react'

interface Props { result: AskResponse }

export default function AnswerCard({ result }: Props) {
  return (
    <div className="animate-fade-in space-y-3">
      {/* Answer */}
      <div className="p-5 rounded-xl bg-bg-card border border-white/[0.06]">
        <div className="flex items-center gap-2 mb-3">
          <CheckCircle2 className="w-4 h-4 text-accent-green" />
          <span className="text-xs font-semibold uppercase tracking-wide text-ink-muted">Answer</span>
        </div>
        <p className="text-sm text-ink-primary leading-relaxed whitespace-pre-wrap">{result.answer}</p>
      </div>

      {/* Citations */}
      {result.cited_sources.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {result.cited_sources.map((src, i) => (
            <span
              key={src}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-accent-blue/10 border border-accent-blue/20 text-[11px] font-medium text-accent-blue"
            >
              <span className="w-4 h-4 rounded-full bg-accent-blue text-white text-[9px] flex items-center justify-center font-bold">
                {i + 1}
              </span>
              <span className="font-mono">{src}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
