import type { PipelineResult, QueryResponse } from "@/types/message"

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000"
const SESSION_KEY = "diabetes-assistant-session-id"

export class ApiError extends Error {}

function readStoredSession(): string | null {
  try {
    return sessionStorage.getItem(SESSION_KEY)
  } catch {
    return null
  }
}

function storeSession(id: string) {
  try {
    sessionStorage.setItem(SESSION_KEY, id)
  } catch {
    return
  }
}

export function getSessionId(): string {
  const existing = readStoredSession()
  if (existing) return existing
  const id = crypto.randomUUID()
  storeSession(id)
  return id
}

export function newSessionId(): string {
  const id = crypto.randomUUID()
  storeSession(id)
  return id
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body?.detail === "string") return body.detail
  } catch {
    return `Request failed (${res.status})`
  }
  return `Request failed (${res.status})`
}

export async function sendMessage(sessionId: string, query: string): Promise<PipelineResult> {
  let res: Response
  try {
    res = await fetch(`${BASE_URL}/v1/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    })
  } catch {
    throw new ApiError("Can't reach the server. Is the backend running?")
  }
  if (!res.ok) throw new ApiError(await errorDetail(res))
  const data: QueryResponse = await res.json()
  return data.result
}

export async function deleteSession(sessionId: string): Promise<void> {
  try {
    await fetch(`${BASE_URL}/v1/sessions/${sessionId}`, { method: "DELETE" })
  } catch {
    return
  }
}
