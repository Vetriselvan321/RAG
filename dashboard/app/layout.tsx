import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Sidebar from '@/components/Sidebar'
import { APIKeyProvider } from '@/context/APIKeyContext'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export const metadata: Metadata = {
  title: 'DocuRetrieve — A Document RAG Engine',
  description:
    'Production-grade Retrieval Augmented Generation with hybrid BM25 + vector search, cross-encoder reranking, and LLM-powered cited answers.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-bg-primary text-ink-primary min-h-screen overflow-hidden">
        <APIKeyProvider>
          <div className="flex h-screen">
            {/* Sidebar */}
            <Sidebar />

            {/* Main column */}
            <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
              {/* Top bar */}
              <header className="h-14 flex-shrink-0 flex items-center justify-between px-6 border-b border-white/[0.06] bg-bg-primary/80 backdrop-blur-sm">
                {/* Left */}
                <div className="hidden sm:flex items-center gap-3">
                  <span className="text-xs text-ink-muted font-medium">
                    ⚡ Enterprise RAG ·{' '}
                    <span className="text-ink-secondary">
                      Hybrid Search · Cross-Encoder Reranking · LLM Citations
                    </span>
                  </span>
                </div>

                {/* Right */}
                <div className="ml-auto flex items-center gap-2">
                  <a
                    href=""
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-ink-muted hover:text-ink-primary hover:bg-white/[0.05] text-xs font-medium transition-all"
                  >
                    <svg
                      viewBox="0 0 24 24"
                      className="w-3.5 h-3.5"
                      fill="currentColor"
                    >
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                    </svg>
                    GitHub
                  </a>

                  <a
                    href={`${API_URL}/docs`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-ink-muted hover:text-ink-primary hover:bg-white/[0.05] text-xs font-medium transition-all"
                  >
                    API Docs
                  </a>
                </div>
              </header>

              {/* Page */}
              <main className="flex-1 overflow-hidden">
                {children}
              </main>
            </div>
          </div>
        </APIKeyProvider>
      </body>
    </html>
  )
}