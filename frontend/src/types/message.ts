export type RiskLevel = "Low" | "Medium" | "High"

export interface RiskAssessment {
  patient_id: string
  risk_level: RiskLevel
  risk_score: number | null
  key_factors: string[]
  reasoning: string
  used_cohort_context: boolean
  model_version: string | null
}

export interface Recommendation {
  patient_id: string
  priority_actions: string[]
  rationale: string
}

export type PipelineResult =
  | { type: "risk_assessment"; assessment: RiskAssessment; from_cache: boolean }
  | { type: "recommendation"; assessment: RiskAssessment; recommendation: Recommendation; from_cache: boolean }
  | { type: "clarification"; message: string }
  | { type: "refusal"; category: string; message: string }

export interface QueryResponse {
  session_id: string
  result: PipelineResult
}

export type ChatMessage =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: PipelineResult }
  | { id: string; role: "error"; content: string; query: string }
