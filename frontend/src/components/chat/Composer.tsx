import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input"

interface ComposerProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  busy: boolean
  autoFocus?: boolean
}

export default function Composer({ value, onChange, onSubmit, busy, autoFocus }: ComposerProps) {
  return (
    <div className="w-full">
      <PromptInput
        onSubmit={onSubmit}
        className="[&_[data-slot=input-group]]:rounded-2xl [&_[data-slot=input-group]]:border-border [&_[data-slot=input-group]]:bg-card [&_[data-slot=input-group]]:shadow-sm dark:[&_[data-slot=input-group]]:bg-card"
      >
        <PromptInputTextarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Ask about a patient's risk or next steps…"
          autoFocus={autoFocus}
          className="min-h-14 px-4 pt-3.5 text-[0.9375rem] leading-relaxed md:text-[0.9375rem] xl:min-h-16 xl:max-h-64 xl:px-5 xl:pt-4 xl:text-base"
        />
        <PromptInputFooter className="justify-end px-2.5 pb-2.5 xl:px-3 xl:pb-3">
          <PromptInputSubmit
            status={busy ? "submitted" : "ready"}
            disabled={!value.trim() || busy}
            className="rounded-full xl:size-9 xl:[&_svg]:size-[1.125rem]"
          />
        </PromptInputFooter>
      </PromptInput>
      <p className="mt-2 text-center text-xs text-muted-foreground">
        Built on a public research dataset. Not medical advice.
      </p>
    </div>
  )
}
