import { ClipboardList, HeartPulse, Stethoscope, Users } from "lucide-react"
import type { LucideIcon } from "lucide-react"

interface Suggestion {
  icon: LucideIcon
  title: string
  prompt: string
}

const SUGGESTIONS: Suggestion[] = [
  { icon: HeartPulse, title: "Assess risk", prompt: "What is patient 0's risk level?" },
  { icon: ClipboardList, title: "Get recommendations", prompt: "What should patient 12 do to lower their risk?" },
  { icon: Users, title: "Compare a profile", prompt: "How risky does patient 5 look?" },
  { icon: Stethoscope, title: "Plan next steps", prompt: "Give patient 42 a prioritized action plan." },
]

export function EmptyGreeting() {
  return (
    <div className="space-y-2 text-center">
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">How can I help today?</h1>
      <p className="text-sm text-muted-foreground">
        Assess a patient's diabetes risk or get lifestyle recommendations targeted to their profile.
      </p>
    </div>
  )
}

export function SuggestionGrid({ onPick, disabled }: { onPick: (prompt: string) => void; disabled: boolean }) {
  return (
    <div className="grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
      {SUGGESTIONS.map(({ icon: Icon, title, prompt }) => (
        <button
          key={title}
          type="button"
          disabled={disabled}
          onClick={() => onPick(prompt)}
          className="group flex items-start gap-3 rounded-xl border border-border bg-card/40 px-3.5 py-3 text-left transition-colors hover:bg-accent disabled:pointer-events-none disabled:opacity-50"
        >
          <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground" />
          <span className="min-w-0">
            <span className="block text-sm font-medium">{title}</span>
            <span className="block truncate text-xs text-muted-foreground">{prompt}</span>
          </span>
        </button>
      ))}
    </div>
  )
}
