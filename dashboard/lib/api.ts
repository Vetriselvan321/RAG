const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

function groqHeaders(apiKey: string) {
  return {
    'Content-Type': 'application/json',
    ...(apiKey ? { 'X-Groq-API-Key': apiKey } : {}),
  }
}

function geminiHeaders(apiKey: string) {
  return {
    ...(apiKey ? { 'X-Gemini-API-Key': apiKey } : {}),
  }
}

export interface SearchResult {
  text: string
  score: number
  source?: string
  metadata?: Record<string, unknown>
}

export interface AskResponse {
  answer: string
  citations?: SearchResult[]
}

export interface UploadResponse {
  message: string
  chunks_created?: number
}

export interface StatsResponse {
  collection_name: string
  points_count: number
  vector_size?: number
}

export async function ask(
  query: string,
  topK = 5,
  groqApiKey = ''
): Promise<AskResponse> {
  const response = await fetch(`${API_URL}/ask`, {
    method: 'POST',
    headers: groqHeaders(groqApiKey),
    body: JSON.stringify({
      query,
      top_k: topK,
    }),
  })

  if (!response.ok) {
    throw new Error(await response.text())
  }

  return response.json()
}

export async function search(
  query: string,
  topK = 5
): Promise<SearchResult[]> {
  const response = await fetch(`${API_URL}/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      top_k: topK,
    }),
  })

  if (!response.ok) {
    throw new Error(await response.text())
  }

  return response.json()
}

export async function uploadDocument(
  file: File
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(await response.text())
  }

  return response.json()
}

export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_URL}/stats`)

  if (!response.ok) {
    throw new Error(await response.text())
  }

  return response.json()
}

export async function* streamAsk(
  query: string,
  topK = 5,
  groqApiKey = '',
): AsyncGenerator<any> {
  const response = await fetch(`${API_URL}/stream`, {
    method: 'POST',
    headers: groqHeaders(groqApiKey),
    body: JSON.stringify({
      query,
      top_k: topK,
    }),
  })

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`

    try {
      const data = await response.json()
      if (data.detail) {
        detail = data.detail
      }
    } catch {
      // Ignore JSON parsing errors
    }

    throw new Error(detail)
  }

  if (!response.body) {
    throw new Error('Streaming is not supported by this response.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()

      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''

      for (const event of events) {
        const line = event
          .split('\n')
          .find(line => line.startsWith('data: '))

        if (!line) continue

        const data = line.slice(6)

        if (data === '[DONE]') {
          return
        }

        try {
          yield JSON.parse(data)
        } catch {
          // Ignore malformed SSE events
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
export async function askImage(
  file: File,
  query?: string,
  topK = 5,
  geminiApiKey = ''
): Promise<any> {
  const formData = new FormData()
  formData.append('file', file)

  if (query) {
    formData.append('query', query)
  }

  formData.append('top_k', String(topK))

  const response = await fetch(`${API_URL}/ask-image`, {
    method: 'POST',
    headers: geminiHeaders(geminiApiKey),
    body: formData,
  })

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`

    try {
      const data = await response.json()
      if (data.detail) {
        detail = data.detail
      }
    } catch {
      // Ignore JSON parsing errors
    }

    throw new Error(detail)
  }

  return response.json()
}