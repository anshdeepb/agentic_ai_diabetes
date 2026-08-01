import { useState } from "react"
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation"
import {
  Message,
  MessageContent,
} from "@/components/ai-elements/message"
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputSubmit,
} from "@/components/ai-elements/prompt-input"
import { sendQuery } from "@/lib/api"
import AssistantResult from "@/components/AssistantResult"
import type { ChatMessage } from "@/types/message"

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [status, setStatus] = useState<"ready" | "streaming">("ready")

  const handleSubmit = async () => {
    if (!input.trim()) return

    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: "user", content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setStatus("streaming")

    const result = await sendQuery(userMessage.content as string)

    const assistantMessage: ChatMessage = { id: crypto.randomUUID(), role: "assistant", content: result }
    setMessages((prev) => [...prev, assistantMessage])
    setStatus("ready")
  }

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <header className="border-b border-border/50 px-6 py-4">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-sm font-medium">Diabetes Risk Assistant</h1>
          <p className="text-xs text-muted-foreground mt-0.5">Ask about a patient's risk or recommendations</p>
        </div>
      </header>

      <div className="flex flex-col flex-1 max-w-2xl w-full mx-auto px-4 overflow-hidden">
        <Conversation className="flex-1">
          <ConversationContent>
            {messages.length === 0 && (
              <div className="h-full flex items-center justify-center">
                <p className="text-sm text-muted-foreground">Try: "What is patient 0's risk level?"</p>
              </div>
            )}
            {messages.map((m) => (
              <Message from={m.role} key={m.id}>
                <MessageContent>
                  {typeof m.content === "string" ? m.content : <AssistantResult result={m.content} />}
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
          <PromptInputSubmit status={status} disabled={!input.trim()} className="mr-1" />
        </PromptInput>
      </div>
    </div>
  )
}