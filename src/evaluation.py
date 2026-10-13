from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

def evaluate_scores(labels: list[int], scores: list[float], threshold: float, latencies_ms: list[float] | None = None) -> dict:
    if not labels: return {"evaluated_images": 0, "message": "No valid test images were available."}
    y, s = np.asarray(labels), np.asarray(scores); pred = (s >= threshold).astype(int)
    p,r,f,_ = precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    result = {"evaluated_images": int(len(y)), "threshold": float(threshold), "precision": float(p), "recall": float(r), "f1": float(f), "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist(), "average_latency_ms": float(np.mean(latencies_ms or [0]))}
    result["roc_auc"] = float(roc_auc_score(y,s)) if len(np.unique(y)) == 2 else None
    return result
def save_results(result: dict, json_path: Path, plot_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True); json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if "confusion_matrix" in result:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,3)); ax.imshow(result["confusion_matrix"], cmap="Blues"); ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"], title="Image-level confusion matrix")
        for i,row in enumerate(result["confusion_matrix"]):
            for j,val in enumerate(row): ax.text(j,i,str(val),ha="center",va="center")
        fig.tight_layout(); fig.savefig(plot_path, dpi=150); plt.close(fig)
