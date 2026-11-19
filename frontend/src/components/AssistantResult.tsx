import { ChevronDown, History, Users } from "lucide-react"
import type { ReactNode } from "react"
import { MessageResponse } from "@/components/ai-elements/message"
import { cn } from "@/lib/utils"
import type { PipelineResult, Recommendation, RiskAssessment, RiskLevel } from "@/types/message"

const RISK_STYLES: Record<RiskLevel, { pill: string; dot: string }> = {
  Low: {
    pill: "bg-emerald-500/10 text-emerald-700 ring-emerald-600/20 dark:text-emerald-400 dark:ring-emerald-400/25",
    dot: "bg-emerald-500",
  },
  Medium: {
    pill: "bg-amber-500/10 text-amber-700 ring-amber-600/20 dark:text-amber-400 dark:ring-amber-400/25",
    dot: "bg-amber-500",
  },
  High: {
    pill: "bg-rose-500/10 text-rose-700 ring-rose-600/20 dark:text-rose-400 dark:ring-rose-400/25",
    dot: "bg-rose-500",
  },
}

function SectionLabel({ children }: { children: ReactNode }) {
  return <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{children}</p>
}

function MetaTag({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground [&_svg]:size-3.5">
      {icon}
      {children}
    </span>
  )
}

function RiskPill({ level, score }: { level: RiskLevel; score: number | null }) {
  const style = RISK_STYLES[level]
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset",
        style.pill
      )}
    >
      <span className={cn("size-1.5 rounded-full", style.dot)} />
      {level} risk
      {score !== null && <span className="font-normal opacity-80 tabular-nums">· {Math.round(score * 100)}%</span>}
    </span>
  )
}

function AssessmentView({ assessment, fromCache }: { assessment: RiskAssessment; fromCache: boolean }) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h3 className="text-base font-semibold tracking-tight">Patient {assessment.patient_id}</h3>
        <RiskPill level={assessment.risk_level} score={assessment.risk_score} />
      </div>

      <p className="text-[15px] leading-relaxed">{assessment.reasoning}</p>

      {assessment.key_factors.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>Key factors</SectionLabel>
          <div className="flex flex-wrap gap-1.5">
            {assessment.key_factors.map((factor) => (
              <span key={factor} className="rounded-md bg-muted px-2 py-1 text-xs text-foreground/90">
                {factor}
              </span>
            ))}
          </div>
        </div>
      )}

      {(assessment.used_cohort_context || fromCache) && (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {assessment.used_cohort_context && <MetaTag icon={<Users />}>Compared with age cohort</MetaTag>}
          {fromCache && <MetaTag icon={<History />}>Reused from earlier in this chat</MetaTag>}
        </div>
      )}

      {assessment.risk_score !== null && (
        <p className="text-xs text-muted-foreground">
          Score from a logistic regression model trained on this dataset
          {assessment.model_version ? ` (${assessment.model_version})` : ""}. A screening estimate, not a diagnosis.
        </p>
      )}
    </div>
  )
}

function RecommendationView({ recommendation }: { recommendation: Recommendation }) {
  return (
    <div className="space-y-4">
      <h3 className="text-base font-semibold tracking-tight">
        Recommendations for patient {recommendation.patient_id}
      </h3>

      <ol className="space-y-3">
        {recommendation.priority_actions.map((action, i) => (
          <li key={action} className="flex gap-3">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium tabular-nums">
              {i + 1}
            </span>
            <span className="pt-0.5 text-[15px] leading-relaxed">{action}</span>
          </li>
        ))}
      </ol>

      <details className="group rounded-xl border border-border">
        <summary className="flex cursor-pointer list-none items-center justify-between px-3.5 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground [&::-webkit-details-marker]:hidden">
          Why these actions
          <ChevronDown className="size-4 transition-transform group-open:rotate-180" />
        </summary>
        <div className="border-t border-border px-3.5 py-3 text-sm leading-relaxed text-foreground/90">
          <MessageResponse>{recommendation.rationale}</MessageResponse>
        </div>
      </details>
    </div>
  )
}

export default function AssistantResult({ result }: { result: PipelineResult }) {
  switch (result.type) {
    case "clarification":
      return <p className="text-[15px] leading-relaxed">{result.message}</p>
    case "risk_assessment":
      return <AssessmentView assessment={result.assessment} fromCache={result.from_cache} />
    case "recommendation":
      return <RecommendationView recommendation={result.recommendation} />
  }
}
