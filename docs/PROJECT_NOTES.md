# VoltVision AI — Project notes

## Summary

VoltVision AI is an independent, educational visual-inspection prototype for cable images. A pretrained ResNet18 serves as a feature extractor, a nearest-neighbour reference bank is built only from normal images, and an image is flagged when its local visual patterns are unusually far from that bank. The system returns a score, a review decision, an approximate heatmap, the closest known-normal example, and latency. It is not electrical safety testing.

## Technical explanation

The method uses transfer learning: ResNet18 is pretrained on ImageNet, so no CNN is trained from scratch. Its `layer3` feature map is a 14×14 grid of patch embeddings, numeric descriptions of local texture and shape. Only normal training images form the reference bank, so the test set stays untouched for honest evaluation. Each query patch is matched to its nearest normal patch by Euclidean distance; larger distances mean more unusual appearance. The 95th percentile of patch distances is the image score, and the patch-distance grid becomes an approximate heatmap. A deterministic subset of normal training images is held out to select a high-percentile threshold, so test labels never tune it. Evaluation uses precision, recall, F1, ROC-AUC where both classes exist, and latency.

## Key concepts

- A CNN learns hierarchical visual features; an embedding is its numeric image description.
- Spatial feature maps retain locations, unlike one global embedding, so they support approximate local anomaly maps.
- Euclidean distance measures straight-line vector separation; cosine distance measures angle. L2-normalized vectors make their rankings closely related.
- Normal-only learning is useful when defects are rare or unknown, but can flag harmless variation.
- Precision is how often alerts are right; recall is how many real anomalies are caught; F1 balances them; ROC-AUC measures score ranking across thresholds.
- Data leakage is using test information to select a model or threshold; it gives overoptimistic results.
- Inference latency here is elapsed image loading, preprocessing, feature extraction, and scoring time.
- A visual anomaly does not prove electrical failure. Electrical testing remains separate.
- MVTec AD is CC BY-NC-SA 4.0; this project needs attribution and non-commercial use compliance.

## Implemented vs future work

Implemented: normal-only patch nearest-neighbour design, deterministic threshold procedure, content-verified reference artifacts, MVTec-compatible scripts, metrics with provenance, four research experiments, a local CSV event log, a Streamlit UI, and a demo-mode fallback. Current results, including how much they vary across data splits, are in the [README](../README.md#results) and [EXPERIMENTS.md](EXPERIMENTS.md). They come from a small single-category benchmark and are not a production claim.

Future work: calibration on representative production imagery, controlled camera and lighting, drift monitoring, integration with inspection hardware, faster closest-reference retrieval, pixel-level evaluation against the ground-truth masks, and integration with electrical testing.

## Frequently asked questions

1. **Why ResNet18?** Small, well-known, ImageNet-pretrained, and CPU-friendly for a prototype.
2. **Why no defect training?** Normal examples are often easier to curate; unseen defects can still be unusual.
3. **Why patch features?** A global embedding can hide small local damage.
4. **Is the score a probability?** No. It is a nearest-normal distance.
5. **How is the cutoff selected?** High percentile of held-out normal training scores, never test labels.
6. **What does the heatmap mean?** Approximate local novelty, not a certified defect mask.
7. **How is leakage avoided?** Only train/good images create the bank, and a normal validation split selects the threshold.
8. **What if illumination changes?** Scores can shift. Experiment 4 measures mild synthetic blur and ±20% brightness changes with the threshold frozen; real camera and lighting variation still needs representative normal data.
9. **How would this scale?** Batch features, compact the reference bank, and use optimized nearest-neighbour search.
10. **Does it prove cable safety?** No; visual inspection complements electrical and compliance tests.
