# Risk Threshold Analysis

Precision/detection rate trade-offs for the OC-Softmax model (selected for deployment). Noise-augmented model data in Appendix A. All metrics computed on the ASVspoof5 eval set (680,774 trials: 138,688 bonafide, 542,086 spoof).

## Metric Definitions

- **Alert precision**: Of all alerts triggered, what percentage are actual deepfakes?
- **Alert detection rate**: Of all actual deepfakes, what percentage does the system catch?
- **FPR** (False positive rate): Of all real calls, what percentage are incorrectly alerted?
- **FNR** (False negative rate): Of all deepfakes, what percentage are missed?

FPR and detection rate are more meaningful for operational planning than raw alert rate, which is inflated by the eval set's 80% spoof composition.

## OC-Softmax Precision / Detection Rate Table

"False Alarms / 1K calls" assumes 1,000 legitimate calls per day. Selected tier thresholds are **bolded**.

Reference point: real calls score 0.786 on average (cosine similarity to the learned "real speech" template). Fake calls score 0.411 on average, but with wide spread (std 0.569) — many fakes score well above the average real call.

| Precision | Threshold | Detection Rate | FPR | FNR | False Alarms / 1K calls |
|-----------|-----------|--------|-----|-----|------------------------|
| 80%* | 0.998 | 96.0% | 93.8% | 4.0% | ~938 |
| 85% | 0.946 | 65.9% | 45.5% | 34.1% | ~455 |
| **90% (MONITOR)** | **0.690** | **52.9%** | **23.0%** | **47.1%** | **~230** |
| 91% | 0.577 | 50.2% | 19.4% | 49.8% | ~194 |
| 92% | 0.453 | 47.4% | 16.1% | 52.6% | ~161 |
| 93% | 0.321 | 44.1% | 13.0% | 55.9% | ~130 |
| 94% | 0.204 | 40.6% | 10.1% | 59.4% | ~101 |
| **95% (HIGH)** | **0.096** | **36.8%** | **7.6%** | **63.2%** | **~76** |
| 96% | -0.007 | 32.4% | 5.3% | 67.6% | ~53 |
| 97% | -0.106 | 27.5% | 3.3% | 72.5% | ~33 |
| 98% | -0.211 | 21.6% | 1.7% | 78.4% | ~17 |
| **99% (CRITICAL)** | **-0.314** | **15.5%** | **0.6%** | **84.5%** | **~6** |

*\*80% precision requires the system to alert on nearly everything (~93-94% FPR). Not viable for any deployment scenario — included only as a lower-bound reference.*

## Selected Configuration: Three-Tier Precision-Anchored

Three severity tiers anchored to precision operating points, plus UNCONFIRMED for samples that don't trigger any tier.

| Tier | Precision | Enterprise Action | Rationale |
|------|-----------|-------------------|-----------|
| **CRITICAL** | 99% | Auto-escalate to SOC, terminate/flag call | Supports automated response; ~6 false alarms/1K calls is low enough for call blocking without human review |
| **HIGH** | 95% | Queue for analyst review within SLA | Primary analyst-facing tier; 19/20 alerts are real deepfakes; ~76/1K false alarms is a manageable queue |
| **MONITOR** | 90% | Log for trend analysis, no immediate action | Passive tier favoring detection rate (~1 in 2 deepfakes caught); higher FPR (~23%) acceptable since no disruptive action is taken |
| **UNCONFIRMED** | — | No action | Not verified safe — 63-67% of deepfakes pass through undetected |

### OC-Softmax Thresholds

| Tier | Threshold | Detection Rate | FPR | False Alarms / 1K calls |
|------|-----------|--------|-----|------------------------|
| **CRITICAL** | score < -0.314 | 15.5% | 0.6% | ~6 |
| **HIGH** | score < 0.096 | 36.8% | 7.6% | ~76 |
| **MONITOR** | score < 0.690 | 52.9% | 23.0% | ~230 |

The trade-off is remarkably linear — each 1% of precision costs about 3-4% of detection rate, with no natural elbow. The viable range for actionable tiers is **93-97%**; below 85%, precision approaches the class prior; above 97%, detection rate drops below 1 in 4.

### Limitations

- **38% EER ceiling**: No threshold can achieve high precision and high detection rate simultaneously. Improving detection rate requires a better model (e.g., SSL front-end), not a different threshold.
- **Class ratio effect**: The eval set is 1:3.9 bonafide:spoof. In real deployment (mostly legitimate calls), FPR and detection rate stay constant but precision improves — the 95% figure is conservative.

---

## Appendix A: Noise-Augmented Model

Included for reference. OC-Softmax v1 was selected for deployment (see `PROJECT.md` — Final Enhanced Model).

### Precision / Detection Rate Table

| Precision | Threshold | Detection Rate | FPR | FNR | False Alarms / 1K calls |
|-----------|-----------|--------|-----|-----|------------------------|
| 80%* | 7.92 | 95.0% | 92.9% | 5.0% | ~929 |
| 85% | -1.66 | 66.7% | 46.0% | 33.3% | ~460 |
| **90% (MONITOR)** | **-3.78** | **51.1%** | **22.2%** | **48.9%** | **~222** |
| 91% | -4.21 | 47.9% | 18.5% | 52.1% | ~185 |
| 92% | -4.66 | 44.6% | 15.2% | 55.4% | ~152 |
| 93% | -5.16 | 41.1% | 12.1% | 58.9% | ~121 |
| 94% | -5.72 | 37.2% | 9.3% | 62.8% | ~93 |
| **95% (HIGH)** | **-6.35** | **33.2%** | **6.8%** | **66.8%** | **~68** |
| 96% | -7.11 | 28.7% | 4.7% | 71.3% | ~47 |
| 97% | -8.06 | 23.7% | 2.9% | 76.3% | ~29 |
| 98% | -9.40 | 17.8% | 1.4% | 82.2% | ~14 |
| **99% (CRITICAL)** | **-11.48** | **11.2%** | **0.4%** | **88.8%** | **~4** |

### Tier Thresholds

| Tier | Threshold | Detection Rate | FPR | False Alarms / 1K calls |
|------|-----------|--------|-----|------------------------|
| **CRITICAL** | score < -11.48 | 11.2% | 0.4% | ~4 |
| **HIGH** | score < -6.35 | 33.2% | 6.8% | ~68 |
| **MONITOR** | score < -3.78 | 51.1% | 22.2% | ~222 |
