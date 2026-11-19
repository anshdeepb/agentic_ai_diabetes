import { useState } from "react"
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation"
import AppHeader from "@/components/chat/AppHeader"
import ChatMessageRow from "@/components/chat/ChatMessageRow"
import Composer from "@/components/chat/Composer"
import { EmptyGreeting, SuggestionGrid } from "@/components/chat/EmptyState"
import ThinkingIndicator from "@/components/chat/ThinkingIndicator"
import { ApiError, deleteSession, getSessionId, newSessionId, sendMessage } from "@/lib/api"
import { useTheme } from "@/lib/theme"
import type { ChatMessage } from "@/types/message"

export default function App() {
  const { theme, toggle } = useTheme()
  const [sessionId, setSessionId] = useState(getSessionId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)

  const ask = async (query: string, { echo = true } = {}) => {
    const text = query.trim()
    if (!text || busy) return

    if (echo) {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: text }])
    }
    setInput("")
    setBusy(true)

    try {
      const result = await sendMessage(sessionId, text)
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "assistant", content: result }])
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Something went wrong. Please try again."
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "error", content: message, query: text }])
    } finally {
      setBusy(false)
    }
  }

  const retry = (errorId: string, query: string) => {
    setMessages((prev) => prev.filter((m) => m.id !== errorId))
    void ask(query, { echo: false })
  }

  const newChat = async () => {
    await deleteSession(sessionId)
    setSessionId(newSessionId())
    setMessages([])
    setInput("")
  }

  const isEmpty = messages.length === 0 && !busy
  const lastId = messages.at(-1)?.id

  return (
    <div className="flex h-dvh flex-col bg-background text-foreground">
      <AppHeader
        theme={theme}
        onToggleTheme={toggle}
        onNewChat={newChat}
        canStartNewChat={messages.length > 0 && !busy}
      />

      {isEmpty ? (
        <main className="flex flex-1 flex-col items-center justify-center px-4 pb-[12vh]">
          <div className="flex w-full max-w-2xl flex-col items-center gap-8">
            <EmptyGreeting />
            <Composer value={input} onChange={setInput} onSubmit={() => ask(input)} busy={busy} autoFocus />
            <SuggestionGrid onPick={(prompt) => ask(prompt)} disabled={busy} />
          </div>
        </main>
      ) : (
        <main className="flex min-h-0 flex-1 flex-col">
          <Conversation className="flex-1">
            <ConversationContent className="mx-auto w-full max-w-2xl gap-8 px-4 py-6">
              {messages.map((m) => (
                <ChatMessageRow
                  key={m.id}
                  message={m}
                  onRetry={m.role === "error" && m.id === lastId && !busy ? () => retry(m.id, m.query) : undefined}
                />
              ))}
              {busy && <ThinkingIndicator />}
            </ConversationContent>
            <ConversationScrollButton />
          </Conversation>
          <div className="mx-auto w-full max-w-2xl px-4 pb-4 pt-2">
            <Composer value={input} onChange={setInput} onSubmit={() => ask(input)} busy={busy} autoFocus />
          </div>
        </main>
      )}
    </div>
  )
}
