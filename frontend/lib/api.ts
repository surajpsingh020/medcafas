import { PredictResponse } from './types';

export async function analyseQuestion(question: string): Promise<PredictResponse> {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail ?? `Server error: ${res.status}`);
  }

  return res.json();
}
