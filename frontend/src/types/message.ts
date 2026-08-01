export type RiskLevel = "Low" | "Medium" | "High"

export interface RiskAssessment {
  patient_id: string
  risk_level: RiskLevel
  key_factors: string[]
  reasoning: string
  used_cohort_context: boolean
}

export interface Recommendation {
  patient_id: string
  priority_actions: string[]
  rationale: string
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string | RiskAssessment | Recommendation
}