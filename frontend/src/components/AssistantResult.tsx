// src/components/AssistantResult.tsx
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { RiskAssessment, Recommendation } from "@/types/message"

function isRiskAssessment(result: RiskAssessment | Recommendation): result is RiskAssessment {
  return "risk_level" in result
}

const RISK_STYLES: Record<RiskAssessment["risk_level"], string> = {
  Low: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  Medium: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  High: "bg-rose-500/15 text-rose-400 border-rose-500/30",
}

export default function AssistantResult({ result }: { result: RiskAssessment | Recommendation }) {
  if (isRiskAssessment(result)) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Patient {result.patient_id}</span>
            <Badge variant="outline" className={RISK_STYLES[result.risk_level]}>
              {result.risk_level} risk
            </Badge>
          </div>
          <ul className="space-y-1">
            {result.key_factors.map((f) => (
              <li key={f} className="text-sm flex gap-2">
                <span className="text-muted-foreground">·</span>
                {f}
              </li>
            ))}
          </ul>
          <p className="text-sm text-muted-foreground border-t border-border/50 pt-3">
            {result.reasoning}
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border/50 bg-card/50">
      <CardContent className="p-4 space-y-3">
        <span className="text-sm text-muted-foreground">Patient {result.patient_id} — recommendations</span>
        <ol className="space-y-1.5">
          {result.priority_actions.map((a, i) => (
            <li key={a} className="text-sm flex gap-2">
              <span className="text-muted-foreground font-mono text-xs mt-0.5">{i + 1}.</span>
              {a}
            </li>
          ))}
        </ol>
        <p className="text-sm text-muted-foreground border-t border-border/50 pt-3">
          {result.rationale}
        </p>
      </CardContent>
    </Card>
  )
}