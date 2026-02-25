'use client';

import { ClaimResult, TemporalFlag } from '@/lib/types';

interface Props {
  claims: ClaimResult[];
  temporalFlags: TemporalFlag[];
}

const VERDICT_CONFIG = {
  SUPPORTED:    { bg: 'bg-green-50',  border: 'border-green-200', badge: 'bg-green-100 text-green-800',  dot: 'bg-green-500' },
  UNSUPPORTED:  { bg: 'bg-yellow-50', border: 'border-yellow-200', badge: 'bg-yellow-100 text-yellow-800', dot: 'bg-yellow-500' },
  CONTRADICTED: { bg: 'bg-red-50',    border: 'border-red-200',    badge: 'bg-red-100 text-red-800',    dot: 'bg-red-500' },
};

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 text-right text-xs text-gray-500">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-gray-100 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="w-8 text-xs font-mono text-gray-600">{value.toFixed(2)}</span>
    </div>
  );
}

export default function ClaimBreakdown({ claims, temporalFlags }: Props) {
  if (!claims.length && !temporalFlags.length) return null;

  return (
    <div className="space-y-4">
      <h2 className="flex items-center gap-2 text-xl font-bold text-gray-800">
        <span>🧩</span>
        Per-Claim Verification
        <span className="ml-auto text-sm font-normal text-gray-400">
          {claims.length} claim{claims.length !== 1 ? 's' : ''} analysed
        </span>
      </h2>

      {/* Temporal flags */}
      {temporalFlags.length > 0 && (
        <div className="rounded-xl border border-orange-200 bg-orange-50 px-5 py-3">
          <p className="mb-1 flex items-center gap-2 text-sm font-semibold text-orange-800">
            <span>🕐</span> Temporal Hallucination Detected
          </p>
          {temporalFlags.map((tf, i) => (
            <p key={i} className="mt-1 text-xs text-orange-700">
              <span className="font-semibold">Year {tf.year}</span> — {tf.reason}
            </p>
          ))}
        </div>
      )}

      {/* Per-claim cards */}
      <div className="space-y-3">
        {claims.map((cr, idx) => {
          const cfg = VERDICT_CONFIG[cr.verdict];
          return (
            <div
              key={idx}
              className={`rounded-xl border p-4 ${cfg.bg} ${cfg.border}`}
            >
              {/* Header row */}
              <div className="flex items-start justify-between gap-3 mb-3">
                <div className="flex items-center gap-2 min-w-0">
                  <span className={`h-2 w-2 rounded-full flex-shrink-0 mt-1 ${cfg.dot}`} />
                  <p className="text-sm font-medium text-gray-800 leading-snug">{cr.claim}</p>
                </div>
                <span className={`flex-shrink-0 rounded-full px-2.5 py-0.5 text-xs font-semibold ${cfg.badge}`}>
                  {cr.verdict}
                </span>
              </div>

              {/* Score bars */}
              <div className="space-y-1 mb-3">
                <ScoreBar label="Entailment" value={cr.entailment} color="bg-green-400" />
                <ScoreBar label="Contradiction" value={cr.contradiction} color="bg-red-400" />
                <ScoreBar label="KB Similarity" value={cr.retrieval_sim} color="bg-blue-400" />
              </div>

              {/* Evidence snippet */}
              {cr.best_evidence && (
                <div className="rounded-lg bg-white/70 border border-gray-200 px-3 py-2">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400 mb-1">
                    Best KB evidence
                  </p>
                  <p className="text-xs text-gray-600 line-clamp-3">{cr.best_evidence}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
