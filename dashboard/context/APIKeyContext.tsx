'use client'

import { createContext, useContext, useEffect, useState } from 'react'

interface APIKeyContextType {
  groqApiKey: string
  geminiApiKey: string
  setGroqApiKey: (key: string) => void
  setGeminiApiKey: (key: string) => void
}

const APIKeyContext = createContext<APIKeyContextType | undefined>(undefined)

export function APIKeyProvider({ children }: { children: React.ReactNode }) {
  const [groqApiKey, setGroqApiKeyState] = useState('')
  const [geminiApiKey, setGeminiApiKeyState] = useState('')

  useEffect(() => {
    const savedGroqKey = sessionStorage.getItem('groq_api_key')
    const savedGeminiKey = sessionStorage.getItem('gemini_api_key')

    if (savedGroqKey) setGroqApiKeyState(savedGroqKey)
    if (savedGeminiKey) setGeminiApiKeyState(savedGeminiKey)
  }, [])

  function setGroqApiKey(key: string) {
    setGroqApiKeyState(key)

    if (key.trim()) {
      sessionStorage.setItem('groq_api_key', key)
    } else {
      sessionStorage.removeItem('groq_api_key')
    }
  }

  function setGeminiApiKey(key: string) {
    setGeminiApiKeyState(key)

    if (key.trim()) {
      sessionStorage.setItem('gemini_api_key', key)
    } else {
      sessionStorage.removeItem('gemini_api_key')
    }
  }

  return (
    <APIKeyContext.Provider
      value={{
        groqApiKey,
        geminiApiKey,
        setGroqApiKey,
        setGeminiApiKey,
      }}
    >
      {children}
    </APIKeyContext.Provider>
  )
}

export function useAPIKeys() {
  const context = useContext(APIKeyContext)

  if (!context) {
    throw new Error('useAPIKeys must be used inside APIKeyProvider')
  }

  return context
}