import { Breakdown } from '@/lib/types';

interface Layer {
  label: string;
  score: number;
  risk: number;
  description: string;
  color: string;
}

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="mt-2 h-2 w-full rounded-full bg-gray-200">
      <div
        className={`h-2 rounded-full transition-all duration-700 ${color}`}
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}

function LayerCard({ layer }: { layer: Layer }) {
  const riskColor =
    layer.risk < 0.35 ? 'text-green-600' : layer.risk < 0.62 ? 'text-yellow-600' : 'text-red-600';

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-2">
        <span className={`h-3 w-3 rounded-full ${layer.color}`} />
        <span className="font-semibold text-gray-700">{layer.label}</span>
      </div>
      <p className="mt-3 text-3xl font-bold text-gray-900">{layer.score.toFixed(2)}</p>
      <ScoreBar value={layer.score} color={layer.color} />
      <p className={`mt-2 text-sm font-medium ${riskColor}`}>
        ↑ risk: {layer.risk.toFixed(2)}
      </p>
      <p className="mt-2 text-xs text-gray-500">{layer.description}</p>
    </div>
  );
}

export default function LayerBreakdown({ breakdown }: { breakdown: Breakdown }) {
  const layers: Layer[] = [
    {
      label: 'Self-Consistency',
      score: breakdown.consistency_score,
      risk: breakdown.consistency_risk,
      description: 'Semantic similarity across 3 independent LLM samples. Low = model is unsure.',
      color: 'bg-blue-500',
    },
    {
      label: 'Evidence Retrieval',
      score: breakdown.retrieval_score,
      risk: breakdown.retrieval_risk,
      description: 'Cosine similarity of answer vs. closest KB document. Low = unsupported claim.',
      color: 'bg-purple-500',
    },
    {
      label: 'NLI Entailment',
      score: breakdown.nli_entailment,
      risk: breakdown.critic_risk,
      description: 'Per-claim cross-encoder entailment: does KB evidence logically support each claim?',
      color: 'bg-orange-500',
    },
  ];

  const vc = breakdown.verdict_counts;

  return (
    <div>
      <h2 className="mb-4 flex items-center gap-2 text-xl font-bold text-gray-800">
        <span>📊</span> Layer-by-Layer Breakdown
      </h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {layers.map((l) => (
          <LayerCard key={l.label} layer={l} />
        ))}
      </div>

      {/* Claim verdict summary */}
      {breakdown.n_claims > 0 && vc && (
        <div className="mt-4 flex flex-wrap items-center gap-4 rounded-xl border border-gray-100 bg-gray-50 px-5 py-3 text-sm">
          <span className="font-semibold text-gray-600">
            {breakdown.n_claims} claim{breakdown.n_claims !== 1 ? 's' : ''}:
          </span>
          <span className="flex items-center gap-1.5 text-green-700">
            <span className="h-2 w-2 rounded-full bg-green-500" />
            {vc.SUPPORTED} supported
          </span>
          <span className="flex items-center gap-1.5 text-yellow-700">
            <span className="h-2 w-2 rounded-full bg-yellow-500" />
            {vc.UNSUPPORTED} unsupported
          </span>
          <span className="flex items-center gap-1.5 text-red-700">
            <span className="h-2 w-2 rounded-full bg-red-500" />
            {vc.CONTRADICTED} contradicted
          </span>
          {breakdown.temporal_risk > 0 && (
            <span className="ml-auto flex items-center gap-1.5 font-semibold text-orange-700">
              🕐 Future-year claim detected
            </span>
          )}
        </div>
      )}
    </div>
  );
}
