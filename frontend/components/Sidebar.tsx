export default function Sidebar() {
  return (
    <aside className="w-72 shrink-0 rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
      <h2 className="mb-4 flex items-center gap-2 text-base font-bold text-gray-800">
        <span className="text-blue-500">ℹ</span> How It Works
      </h2>
      <p className="mb-5 text-sm font-semibold text-gray-600">3-Layer Detection Engine:</p>

      <div className="space-y-5">
        <div>
          <p className="flex items-center gap-1.5 text-sm font-bold text-gray-800">
            <span className="h-3 w-3 rounded-full bg-blue-500" />
            Layer 1 — Self-Consistency
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Asks the LLM the same question 3× at different temperatures and measures
            semantic drift. High drift = model unsure.
          </p>
        </div>

        <div>
          <p className="flex items-center gap-1.5 text-sm font-bold text-gray-800">
            <span className="h-3 w-3 rounded-full bg-purple-500" />
            Layer 2 — Retrieval
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Embeds the answer and searches a medical knowledge base (HaluEval).
            Low similarity = unsupported claim.
          </p>
        </div>

        <div>
          <p className="flex items-center gap-1.5 text-sm font-bold text-gray-800">
            <span className="h-3 w-3 rounded-full bg-orange-500" />
            Layer 3 — NLI Critic
          </p>
          <p className="mt-1 text-xs text-gray-500">
            A cross-encoder model checks if the answer is{' '}
            <em>entailed</em> by the retrieved evidence. Low entailment =
            hallucination risk.
          </p>
        </div>
      </div>

      <div className="mt-6 border-t border-gray-100 pt-5 space-y-2 text-xs text-gray-400">
        <p>⚡ CPU-only · No GPU required</p>
        <p>🏥 Built on HaluEval KB · 2 000 docs</p>
        <p>🔒 Fully local · No data leaves your machine</p>
      </div>
    </aside>
  );
}
