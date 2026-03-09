export interface Citation {
  source: string;
  question: string;
  answer: string;
  similarity: number;
}

export interface ClaimResult {
  claim: string;
  best_evidence: string;
  entailment: number;
  contradiction?: number;
  retrieval_sim: number;
  verdict: 'SUPPORTED' | 'UNSUPPORTED' | 'CONTRADICTED';
}

export interface TemporalFlag {
  claim: string;
  year: number;
  reason: string;
}

export interface VerdictCounts {
  SUPPORTED: number;
  UNSUPPORTED: number;
  CONTRADICTED: number;
}

export interface Breakdown {
  consistency_score: number;
  retrieval_score: number;
  nli_entailment: number;
  consistency_risk: number;
  retrieval_risk: number;
  critic_risk: number;
  temporal_risk: number;
  n_claims: number;
  verdict_counts: VerdictCounts;
}

export type RiskFlag = 'LOW' | 'CAUTION' | 'HIGH';

export interface PredictResponse {
  question: string;
  answer: string;
  risk_flag: RiskFlag;
  risk_score: number;
  confidence: number;
  explanation: string;
  breakdown: Breakdown;
  citations: Citation[];
  claim_breakdown: ClaimResult[];
  temporal_flags: TemporalFlag[];
  latency_ms: number;
}
