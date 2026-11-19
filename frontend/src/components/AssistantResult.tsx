import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { PipelineResult, Recommendation, RiskAssessment } from "@/types/message"

const RISK_STYLES: Record<RiskAssessment["risk_level"], string> = {
  Low: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  Medium: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  High: "bg-rose-500/15 text-rose-400 border-rose-500/30",
}

function AssessmentCard({ assessment, fromCache }: { assessment: RiskAssessment; fromCache: boolean }) {
  return (
    <Card className="border-border/50 bg-card/50">
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm text-muted-foreground">Patient {assessment.patient_id}</span>
          <div className="flex items-center gap-2">
            {fromCache && (
              <Badge variant="outline" className="text-muted-foreground">
                from memory
              </Badge>
            )}
            <Badge variant="outline" className={RISK_STYLES[assessment.risk_level]}>
              {assessment.risk_level} risk
            </Badge>
          </div>
        </div>
        <ul className="space-y-1">
          {assessment.key_factors.map((f) => (
            <li key={f} className="text-sm flex gap-2">
              <span className="text-muted-foreground">·</span>
              {f}
            </li>
          ))}
        </ul>
        <p className="text-sm text-muted-foreground border-t border-border/50 pt-3">{assessment.reasoning}</p>
      </CardContent>
    </Card>
  )
}

function RecommendationCard({ recommendation }: { recommendation: Recommendation }) {
  return (
    <Card className="border-border/50 bg-card/50">
      <CardContent className="p-4 space-y-3">
        <span className="text-sm text-muted-foreground">
          Patient {recommendation.patient_id} — recommendations
        </span>
        <ol className="space-y-1.5">
          {recommendation.priority_actions.map((a, i) => (
            <li key={a} className="text-sm flex gap-2">
              <span className="text-muted-foreground font-mono text-xs mt-0.5">{i + 1}.</span>
              {a}
            </li>
          ))}
        </ol>
        <p className="text-sm text-muted-foreground border-t border-border/50 pt-3">{recommendation.rationale}</p>
      </CardContent>
    </Card>
  )
}

export default function AssistantResult({ result }: { result: PipelineResult }) {
  switch (result.type) {
    case "clarification":
      return <p className="text-sm">{result.message}</p>
    case "risk_assessment":
      return <AssessmentCard assessment={result.assessment} fromCache={result.from_cache} />
    case "recommendation":
      return <RecommendationCard recommendation={result.recommendation} />

  }
}
