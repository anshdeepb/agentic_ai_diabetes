import BrandMark from "@/components/chat/BrandMark"

export default function ThinkingIndicator() {
  return (
    <div className="flex items-end gap-3" role="status" aria-live="polite">
      <BrandMark />
      <div className="flex h-9 items-center gap-1 rounded-2xl rounded-bl-md bg-secondary px-4">
        <span className="typing-dot" />
        <span className="typing-dot [animation-delay:150ms]" />
        <span className="typing-dot [animation-delay:300ms]" />
        <span className="sr-only">Assistant is responding</span>
      </div>
    </div>
  )
}
