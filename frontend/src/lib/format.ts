import type { PipelineResult } from "@/types/message"

export function resultToText(result: PipelineResult): string {
  switch (result.type) {
    case "clarification":
      return result.message
    case "risk_assessment": {
      const a = result.assessment
      return [
        `Patient ${a.patient_id}: ${a.risk_level} risk`,
        "",
        "Key factors:",
        ...a.key_factors.map((f) => `- ${f}`),
        "",
        a.reasoning,
      ].join("\n")
    }
    case "recommendation": {
      const r = result.recommendation
      return [
        `Recommendations for patient ${r.patient_id}`,
        "",
        ...r.priority_actions.map((action, i) => `${i + 1}. ${action}`),
        "",
        r.rationale,
      ].join("\n")
    }
  }
}
