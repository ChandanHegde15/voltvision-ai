# VoltVision AI — interview notes

## 30-second introduction

VoltVision AI is an educational visual-inspection prototype for cable images. I use a pretrained ResNet18 as a feature extractor, build a nearest-neighbor reference bank only from normal images, then flag an uploaded image when its local visual patterns are unusually far from that bank. It provides a score, a review decision, approximate heatmap, and latency; it is not electrical safety testing.

## Two-minute technical explanation

I use transfer learning: ResNet18 is pretrained on ImageNet, so I do not train a CNN from scratch. Its late convolutional feature map is a grid of patch embeddings—numeric descriptions of local texture and shape. Normal train images alone form the reference bank, preserving the test set for honest evaluation. For each query patch I find its nearest normal patch with Euclidean distance. Larger distances mean more unusual appearance. The 95th percentile of patch distances is the image score, and the patch-distance grid becomes an approximate heatmap. I reserve a deterministic subset of normal training data to select a high-percentile threshold; test labels never tune it. Evaluation uses precision, recall, F1, ROC-AUC where both classes exist, and latency.

## Key concepts

- A CNN learns hierarchical visual features; an embedding is its numeric image description.
- Spatial feature maps retain locations, unlike one global embedding, so they support approximate local anomaly maps.
- Euclidean distance measures straight-line vector separation; cosine distance measures angle. L2-normalized vectors make their rankings closely related.
- Normal-only learning is useful when defects are rare or unknown, but can flag harmless variation.
- Precision is how often alerts are right; recall is how many real anomalies are caught; F1 balances them; ROC-AUC measures score ranking across thresholds.
- Data leakage is using test information to select a model or threshold; it gives overoptimistic results.
- Inference latency is elapsed preprocessing, feature extraction, and scoring time.
- A visual anomaly does not prove electrical failure. Electrical testing remains separate.
- MVTec AD is CC BY-NC-SA 4.0; this project needs attribution and non-commercial use compliance.

## Implemented vs future work

Implemented: normal-only patch nearest-neighbor design, deterministic threshold procedure, MVTec-compatible scripts, metrics, local CSV event utility, Streamlit UI, and demo-mode fallback. On the local MVTec cable run, the system used 224 normal training images, a 179/45 reference-validation split, and 150 held-out test images. It measured precision 1.0000, recall 0.8261, F1 0.9048, and ROC-AUC 0.9936; this is a small single-category benchmark, not a production claim. Future work: production calibration, camera controls, drift monitoring, PLC/MQTT integration, model validation on representative factory data, and electrical-test integration.

## Likely questions

1. **Why ResNet18?** Small, well-known, ImageNet-pretrained, and CPU-friendly for an MVP.
2. **Why no defect training?** Normal examples are often easier to curate; unseen defects can still be unusual.
3. **Why patch features?** A global embedding can hide small local damage.
4. **Is the score a probability?** No. It is a nearest-normal distance.
5. **How select the cutoff?** High percentile of held-out normal training scores, never test labels.
6. **What does the heatmap mean?** Approximate local novelty, not a certified defect mask.
7. **How avoid leakage?** Only train/good images create the bank and normal validation selects threshold.
8. **What if illumination changes?** It can raise scores; use controlled acquisition and representative normal data.
9. **How would this scale?** Batch features, compact the reference bank, and use optimized nearest-neighbor search.
10. **Does it prove cable safety?** No; visual inspection complements electrical and compliance tests.
