'use client'

import { useState } from 'react'
import { KeyRound, Eye, EyeOff, Save, Trash2 } from 'lucide-react'
import { useAPIKeys } from '@/context/APIKeyContext'

export default function SettingsPage() {
  const {
    groqApiKey,
    geminiApiKey,
    setGroqApiKey,
    setGeminiApiKey,
  } = useAPIKeys()

  const [showGroq, setShowGroq] = useState(false)
  const [showGemini, setShowGemini] = useState(false)

  const [groqInput, setGroqInput] = useState(groqApiKey)
  const [geminiInput, setGeminiInput] = useState(geminiApiKey)

  function saveKeys() {
    setGroqApiKey(groqInput.trim())
    setGeminiApiKey(geminiInput.trim())
  }

  function clearKeys() {
    setGroqInput('')
    setGeminiInput('')
    setGroqApiKey('')
    setGeminiApiKey('')
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto px-6 py-10">

        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-accent-purple/10 border border-accent-purple/20 flex items-center justify-center">
            <KeyRound className="w-5 h-5 text-accent-purple" />
          </div>

          <h1 className="text-2xl font-bold text-ink-primary">
            API Keys
          </h1>
        </div>

        <p className="text-sm text-ink-muted mb-8">
          Add your own API keys to use the RAG engine. Your keys are kept
          in this browser session and are sent only when making requests.
        </p>

        {/* Groq */}
        <div className="p-5 rounded-2xl bg-bg-card border border-white/[0.06] mb-5">
          <h2 className="text-sm font-semibold text-ink-primary mb-2">
            Groq API Key
          </h2>

          <p className="text-xs text-ink-muted mb-4">
            Used for normal text questions and streaming answers.
          </p>

          <div className="relative">
            <input
              type={showGroq ? 'text' : 'password'}
              value={groqInput}
              onChange={e => setGroqInput(e.target.value)}
              placeholder="gsk_..."
              className="w-full bg-bg-primary border border-white/[0.08] rounded-xl px-4 py-3 pr-12 text-sm text-ink-primary outline-none focus:border-accent-purple/50"
            />

            <button
              type="button"
              onClick={() => setShowGroq(!showGroq)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-muted hover:text-ink-primary"
            >
              {showGroq ? (
                <EyeOff className="w-4 h-4" />
              ) : (
                <Eye className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        {/* Gemini */}
        <div className="p-5 rounded-2xl bg-bg-card border border-white/[0.06] mb-5">
          <h2 className="text-sm font-semibold text-ink-primary mb-2">
            Gemini API Key
          </h2>

          <p className="text-xs text-ink-muted mb-4">
            Used for image understanding and vision questions.
          </p>

          <div className="relative">
            <input
              type={showGemini ? 'text' : 'password'}
              value={geminiInput}
              onChange={e => setGeminiInput(e.target.value)}
              placeholder="AIza..."
              className="w-full bg-bg-primary border border-white/[0.08] rounded-xl px-4 py-3 pr-12 text-sm text-ink-primary outline-none focus:border-accent-purple/50"
            />

            <button
              type="button"
              onClick={() => setShowGemini(!showGemini)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-muted hover:text-ink-primary"
            >
              {showGemini ? (
                <EyeOff className="w-4 h-4" />
              ) : (
                <Eye className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex gap-3">
          <button
            onClick={saveKeys}
            className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-accent-purple text-white text-sm font-semibold hover:opacity-90 transition-all"
          >
            <Save className="w-4 h-4" />
            Save Keys
          </button>

          <button
            onClick={clearKeys}
            className="px-5 flex items-center justify-center gap-2 py-3 rounded-xl border border-white/[0.08] text-ink-muted text-sm hover:text-accent-red hover:border-accent-red/30 transition-all"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        </div>

        <div className="mt-6 p-4 rounded-xl bg-bg-elevated border border-white/[0.05]">
          <p className="text-xs text-ink-muted leading-relaxed">
            <strong className="text-ink-secondary">
              Security note:
            </strong>{' '}
            This application does not need to permanently store your API
            keys. They are kept in your browser session and will be cleared
            when the session storage is cleared.
          </p>
        </div>

      </div>
    </div>
  )
}