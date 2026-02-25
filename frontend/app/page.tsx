'use client';

import { useState, useRef } from 'react';
import { analyseQuestion } from '@/lib/api';
import { PredictResponse } from '@/lib/types';
import RiskBadge from '@/components/RiskBadge';
import LayerBreakdown from '@/components/LayerBreakdown';
import Citations from '@/components/Citations';
import ClaimBreakdown from '@/components/ClaimBreakdown';
import Sidebar from '@/components/Sidebar';

const EXAMPLES = [
  'What are the symptoms of appendicitis?',
  'Ibuprofen and warfarin have no clinically significant interaction.',
  'What were the results of the HORIZON-9 trial published in 2027?',
  'What is the mechanism of action of metformin?',
];

export default function Home() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  async function handleSubmit(q?: string) {
    const query = (q ?? question).trim();
    if (!query) return;
    if (q) setQuestion(q);

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await analyseQuestion(query);
      setResult(res);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white shadow-sm">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-6 py-4">
          <span className="text-3xl">🏥</span>
          <div>
            <h1 className="text-2xl font-extrabold tracking-tight text-gray-900">MedCAFAS</h1>
            <p className="text-sm text-gray-500">
              <span className="font-medium">Medical Confidence-Aware Factuality Assessment System</span>
              {' '}—{' '}
              <em>An ML system that knows when it&apos;s lying</em>
            </p>
          </div>
        </div>
      </header>

      {/* Body */}
      <div className="mx-auto flex w-full max-w-7xl flex-1 gap-8 px-6 py-8">
        {/* Sidebar */}
        <Sidebar />

        {/* Main content */}
        <main className="flex-1 space-y-6">
          {/* Input card */}
          <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
            <label className="mb-2 block text-sm font-semibold text-gray-700">
              Enter a medical question or claim:
            </label>
            <textarea
              className="w-full resize-none rounded-xl border border-gray-300 bg-gray-50 px-4 py-3 text-sm text-gray-900 shadow-inner outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
              rows={4}
              placeholder="e.g. What are the symptoms of appendicitis?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit();
              }}
            />

            <div className="mt-3 flex flex-wrap items-center gap-3">
              <button
                onClick={() => handleSubmit()}
                disabled={loading || !question.trim()}
                className="flex items-center gap-2 rounded-xl bg-red-600 px-6 py-2.5 text-sm font-semibold text-white shadow transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Analysing…
                  </>
                ) : (
                  <>🔬 Analyse</>
                )}
              </button>
              <span className="text-xs text-gray-400">or Ctrl+Enter</span>
            </div>

            {/* Example pills */}
            <div className="mt-4">
              <p className="mb-2 text-xs font-medium text-gray-400 uppercase tracking-wide">Try an example:</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => handleSubmit(ex)}
                    disabled={loading}
                    className="rounded-full border border-gray-300 bg-white px-3 py-1 text-xs text-gray-600 transition hover:border-indigo-400 hover:text-indigo-600 disabled:opacity-40"
                  >
                    {ex.length > 55 ? ex.slice(0, 55) + '…' : ex}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
              ⚠ {error}
            </div>
          )}

          {/* Loading skeleton */}
          {loading && (
            <div className="space-y-4 animate-pulse">
              <div className="h-16 rounded-xl bg-gray-200" />
              <div className="grid grid-cols-3 gap-4">
                {[0, 1, 2].map((i) => <div key={i} className="h-36 rounded-xl bg-gray-200" />)}
              </div>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <div ref={resultRef} className="space-y-6">
              {/* LLM Answer */}
              <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
                <h2 className="mb-3 flex items-center gap-2 text-xl font-bold text-gray-800">
                  <span>📝</span> LLM Answer
                </h2>
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-gray-700">
                  {result.answer}
                </p>
                <p className="mt-3 text-right text-xs text-gray-400">
                  ⏱ Latency: {result.latency_ms.toLocaleString()} ms
                </p>
              </div>

              {/* Risk badge */}
              <RiskBadge
                flag={result.risk_flag}
                score={result.risk_score}
                confidence={result.confidence}
              />

              {/* Explanation */}
              <div className="rounded-xl border border-gray-200 bg-white px-5 py-3 text-sm text-gray-700 shadow-sm">
                {result.explanation}
              </div>

              {/* Layer breakdown */}
              <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
                <LayerBreakdown breakdown={result.breakdown} />
              </div>

              {/* Citations */}
              {result.citations.length > 0 && (
                <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
                  <Citations citations={result.citations} />
                </div>
              )}

              {/* Per-claim FACTSCORE-style breakdown */}
              {(result.claim_breakdown.length > 0 || result.temporal_flags.length > 0) && (
                <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
                  <ClaimBreakdown
                    claims={result.claim_breakdown}
                    temporalFlags={result.temporal_flags}
                  />
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
