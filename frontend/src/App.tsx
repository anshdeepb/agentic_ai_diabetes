import { useState } from "react"
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation"
import { Message, MessageContent } from "@/components/ai-elements/message"
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputSubmit,
} from "@/components/ai-elements/prompt-input"
import { Button } from "@/components/ui/button"
import { ApiError, deleteSession, getSessionId, newSessionId, sendMessage } from "@/lib/api"
import AssistantResult from "@/components/AssistantResult"
import type { ChatMessage } from "@/types/message"

export default function App() {
  const [sessionId, setSessionId] = useState(getSessionId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [status, setStatus] = useState<"ready" | "submitted" | "error">("ready")

  const handleSubmit = async () => {
    const query = input.trim()
    if (!query || status === "submitted") return

    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: query }])
    setInput("")
    setStatus("submitted")

    try {
      const result = await sendMessage(sessionId, query)
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "assistant", content: result }])
      setStatus("ready")
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Something went wrong. Please try again."
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "error", content: message }])
      setStatus("error")
    }
  }

  const handleNewChat = async () => {
    await deleteSession(sessionId)
    setSessionId(newSessionId())
    setMessages([])
    setStatus("ready")
  }

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <header className="border-b border-border/50 px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center justify-between gap-4">
          <div>
            <h1 className="text-sm font-medium">Diabetes Risk Assistant</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Ask about a patient's risk or recommendations</p>
          </div>
          <Button variant="outline" size="sm" onClick={handleNewChat} disabled={status === "submitted"}>
            New chat
          </Button>
        </div>
      </header>

      <div className="flex flex-col flex-1 max-w-2xl w-full mx-auto px-4 overflow-hidden">
        <Conversation className="flex-1">
          <ConversationContent>
            {messages.length === 0 && (
              <div className="h-full flex items-center justify-center">
                <p className="text-sm text-muted-foreground">
                  Try: "What is patient 0's risk level?" then "What should they do about it?"
                </p>
              </div>
            )}
            {messages.map((m) => (
              <Message from={m.role === "user" ? "user" : "assistant"} key={m.id}>
                <MessageContent>
                  {m.role === "user" && m.content}
                  {m.role === "assistant" && <AssistantResult result={m.content} />}
                  {m.role === "error" && <p className="text-sm text-rose-400">{m.content}</p>}
                </MessageContent>
              </Message>
            ))}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <PromptInput onSubmit={handleSubmit} className="mb-4 p-2 gap-2 focus-within:ring-0 focus-within:border-input">
          <PromptInputTextarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a patient..."
          />
          <PromptInputSubmit status={status} disabled={!input.trim() || status === "submitted"} className="mr-1" />
        </PromptInput>
      </div>
    </div>
  )
}
