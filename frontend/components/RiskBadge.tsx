import { RiskFlag } from '@/lib/types';

const cfg = {
  LOW:     { bg: 'bg-green-50',  border: 'border-green-500',  text: 'text-green-800',  dot: 'bg-green-500',  label: 'LOW RISK'     },
  CAUTION: { bg: 'bg-yellow-50', border: 'border-yellow-500', text: 'text-yellow-800', dot: 'bg-yellow-500', label: 'CAUTION'       },
  HIGH:    { bg: 'bg-red-50',    border: 'border-red-500',    text: 'text-red-800',    dot: 'bg-red-500',    label: 'HIGH RISK'    },
};

export default function RiskBadge({
  flag, score, confidence,
}: { flag: RiskFlag; score: number; confidence: number }) {
  const c = cfg[flag];
  return (
    <div className={`flex items-center gap-4 rounded-xl border-l-4 px-6 py-4 ${c.bg} ${c.border}`}>
      <span className={`h-4 w-4 rounded-full ${c.dot} shrink-0 shadow-sm`} />
      <div className="flex flex-wrap gap-6">
        <span className={`text-lg font-bold tracking-wide ${c.text}`}>Risk: {c.label}</span>
        <span className={`text-lg font-semibold ${c.text}`}>Score: {score.toFixed(2)}</span>
        <span className={`text-lg font-semibold ${c.text}`}>Confidence: {confidence.toFixed(2)}</span>
      </div>
    </div>
  );
}
