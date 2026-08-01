import type { RiskAssessment, Recommendation } from "@/types/message"

const BASE_URL = "http://localhost:8000"

export async function sendQuery(query: string): Promise<RiskAssessment | Recommendation> {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  })
  return res.json()
}