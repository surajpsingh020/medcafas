import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

with open("data/eval_llm_final.json", "r") as f:
    data = json.load(f)

# The new weights we tested
OLD_W = {"consistency": 0.25, "retrieval": 0.30, "critic": 0.30, "entity": 0.15}
NEW_W = {"consistency": 0.15, "retrieval": 0.20, "critic": 0.55, "entity": 0.10}

for r in data:
    old_ret_risk = 1.0 - r['retrieval']
    old_nli_risk = 1.0 - r['nli_critic']
    old_ent_risk = r['entity_risk']
    
    known_risk_sum = (OLD_W['retrieval'] * old_ret_risk) + (OLD_W['critic'] * old_nli_risk) + (OLD_W['entity'] * old_ent_risk)
    cons_risk = (r['risk_score'] - known_risk_sum) / OLD_W['consistency']
    cons_risk = max(0.0, min(1.0, cons_risk))
    
    r['new_risk'] = (
        (NEW_W['consistency'] * cons_risk) + 
        (NEW_W['retrieval'] * old_ret_risk) + 
        (NEW_W['critic'] * old_nli_risk) + 
        (NEW_W['entity'] * old_ent_risk)
    )

best_f1 = 0
best_thresh = 0

print("Sweeping RISK_HIGH thresholds to find the Goldilocks Zone...")
for thresh in np.arange(0.30, 0.60, 0.01):
    y_true = [int(r['is_hallucinated']) for r in data]
    y_pred = [int(r['new_risk'] >= thresh) for r in data]
    
    # We optimize for Macro F1 to balance both classes equally
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\n🏆 Best RISK_HIGH Threshold: {best_thresh:.2f}")

y_pred = [int(r['new_risk'] >= best_thresh) for r in data]
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"🚀 Optimized Accuracy: {acc:.1%}")
print("\nConfusion Matrix:")
print(f"                 Pred: NOT-HALL   Pred: HALL")
print(f" True: NOT-HALL       {cm[0][0]:>6}        {cm[0][1]:>6}")
print(f" True: HALL           {cm[1][0]:>6}        {cm[1][1]:>6}\n")
print(classification_report(y_true, y_pred, target_names=["NOT hallucinated", "Hallucinated"]))