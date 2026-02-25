import { Citation } from '@/lib/types';

export default function Citations({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;

  return (
    <div>
      <h2 className="mb-4 flex items-center gap-2 text-xl font-bold text-gray-800">
        <span>📚</span> Knowledge Base Citations
      </h2>
      <div className="space-y-3">
        {citations.map((c, i) => (
          <div
            key={i}
            className="rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm"
          >
            <div className="mb-1 flex items-center justify-between">
              <span className="font-semibold text-gray-600">{c.source}</span>
              <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
                sim: {c.similarity.toFixed(3)}
              </span>
            </div>
            {c.question && (
              <p className="mb-1 text-gray-500 italic">&ldquo;{c.question}&rdquo;</p>
            )}
            <p className="text-gray-700">{c.answer}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
