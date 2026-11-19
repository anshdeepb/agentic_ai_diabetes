import { Check, CircleAlert, Copy, RotateCcw } from "lucide-react"
import { useState } from "react"
import { Message, MessageContent } from "@/components/ai-elements/message"
import { Button } from "@/components/ui/button"
import AssistantResult from "@/components/AssistantResult"
import BrandMark from "@/components/chat/BrandMark"
import { resultToText } from "@/lib/format"
import type { ChatMessage, PipelineResult } from "@/types/message"

function CopyButton({ result }: { result: PipelineResult }) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(resultToText(result))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <Button
      variant="ghost"
      size="icon-xs"
      onClick={copy}
      aria-label={copied ? "Copied" : "Copy response"}
      className="text-muted-foreground hover:text-foreground"
    >
      {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
    </Button>
  )
}

interface ChatMessageRowProps {
  message: ChatMessage
  onRetry?: () => void
}

export default function ChatMessageRow({ message, onRetry }: ChatMessageRowProps) {
  if (message.role === "user") {
    return (
      <Message from="user">
        <MessageContent className="group-[.is-user]:rounded-2xl group-[.is-user]:px-4 group-[.is-user]:py-2.5 text-[0.9375rem] leading-relaxed whitespace-pre-wrap">
          {message.content}
        </MessageContent>
      </Message>
    )
  }

  return (
    <div className="group/row flex gap-3">
      <BrandMark />
      <div className="min-w-0 flex-1 pt-0.5">
        {message.role === "assistant" ? (
          <>
            <AssistantResult result={message.content} />
            <div className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover/row:opacity-100 focus-within:opacity-100">
              <CopyButton result={message.content} />
            </div>
          </>
        ) : (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <span className="inline-flex items-center gap-2 text-sm text-destructive">
              <CircleAlert className="size-4 shrink-0" />
              {message.content}
            </span>
            {onRetry && (
              <Button variant="outline" size="xs" onClick={onRetry} className="gap-1.5">
                <RotateCcw className="size-3" />
                Retry
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
