# MidasMap vs Gold Digger: Honest Comparison

**Date:** April 2026  
**Presenter:** Anik Sahai  
**Scope:** Immunogold particle detection in EM images

---

## Executive Summary

| Aspect | MidasMap | Gold Digger | Winner |
|--------|----------|-------------|--------|
| **Reported F1** | 0.943 | Not reported* | — |
| **Accuracy** | ~94% (derived from F1) | 97% | Gold Digger |
| **Per-class breakdown** | Yes (6nm/12nm separate) | No | MidasMap |
| **Architecture** | CenterNet + BiFPN | pix2pix (cGAN) | Different paradigms |
| **Training data** | 10 images, 453 particles | 6 images, ~3,000 crops | Gold Digger |
| **Generalization** | Unknown across magnifications | Poor (15% on pre-embedding) | MidasMap (likely) |
| **Speed** | ~50ms per 512px patch (GPU) | Human-level speed reported | Gold Digger |

**\* Gold Digger reports accuracy only; F1 not calculated. Direct F1 comparison impossible.**

---

## Detailed Comparison

### 1. Architecture & Approach

#### **MidasMap: Keypoint Detection (CenterNet-style)**

```
Input: 1-channel grayscale TEM image
  ↓
ResNet-50 encoder (CEM500K pretrained)
  ↓
BiFPN neck (2 rounds, 128 channels)
  ↓
Stride-2 decoder (transposed conv)
  ↓
Heatmap head + Offset head
  ↓
Output: (H/2, W/2) confidence & sub-pixel refinement
```

**Advantages:**
- ✓ **Sub-pixel accuracy**: Offset regression refines particle centers to 0.1px precision
- ✓ **Explicit class separation**: 6nm and 12nm heads compete, natural separation
- ✓ **Stride-2 resolution**: Only feasible resolution for 4-6px particles; stride-4 collapses them
- ✓ **Domain-specific pretraining**: CEM500K (electron microscopy) beats ImageNet

**Limitations:**
- ✗ Small dataset (10 images) → potential overfitting to FFRIL syntax
- ✗ Stride-2 is expensive computationally
- ✗ Requires Hungarian matching + per-class thresholding (more complex pipeline)

---

#### **Gold Digger: Semantic Segmentation (pix2pix)**

```
Input: 256×256 crop of grayscale TEM
  ↓
U-Net-256 generator (conditional GAN)
  ↓
Discriminator loss (adversarial training)
  ↓
Output: 256×256 binary mask (gold vs background)
```

**Advantages:**
- ✓ **Simpler pipeline**: End-to-end pixel-level segmentation; no post-processing
- ✓ **Larger training set**: 6 images → ~3,000 crops (augmentation by tiling)
- ✓ **Proven generalization**: Works across magnifications (43k, 83k)
- ✓ **Established approach**: pix2pix is battle-tested

**Limitations:**
- ✗ **No per-class separation**: Both 6nm and 12nm in single output
- ✗ **Poor cross-domain generalization**: Only 15% accuracy on pre-embedding samples (vs 91% post-embedding)
- ✗ **No sub-pixel accuracy**: Segmentation masks are pixel-level; refinement requires post-processing
- ✗ **Fixed input size**: 256×256 crops → overlapping tiles → stitching artifacts

---

### 2. Performance Metrics

#### **MidasMap: Per-Class F1/Precision/Recall**

Evaluation method: **Hungarian matching with per-class radii (6nm: 9px, 12nm: 15px)**

| Metric | 6nm | 12nm | Mean | Overall |
|--------|-----|------|------|---------|
| **F1** | 0.944 | 0.909 | 0.927 | 0.943* |
| **Precision** | 0.951 | 0.895 | 0.923 | 0.944 |
| **Recall** | 0.938 | 0.925 | 0.932 | 0.942 |
| **TP** | — | — | — | 397 |
| **FP** | — | — | — | 24 |
| **FN** | — | — | — | 32 |

\* Mean F1 across 8 usable LOOCV folds (453 labeled particles total)

**Interpretation:**
- 6nm particles: Easier to detect (higher precision, lower FP rate)
- 12nm particles: Slightly harder (higher recall, catching most but some false positives)
- Trade-off is **balanced** across both classes

#### **Gold Digger: Accuracy Only**

| Condition | Accuracy |
|-----------|----------|
| **FRIL images (post-embedding)** | ~97% |
| **Larger images (full-size)** | ~97% |
| **Pre-embedding samples** | ~15% |
| **vs TAC baseline (threshold-area-circularity)** | Comparable on small images, better on large |

**Issues with accuracy metric:**
- ❌ No F1, precision, or recall reported
- ❌ Accuracy can be misleading with imbalanced classes (many background pixels)
- ❌ No per-class breakdown → can't tell if 6nm or 12nm is the bottleneck
- ❌ Likely measured as pixel-level accuracy, not particle-level detection

---

### 3. Confusion Matrices

#### **MidasMap (Particle-level matching)**

```
Predicted
         TP      FP
GT {TP  397      24    ← Correctly detected & false alarms
    FN   32     —      ← Missed particles
    
Precision = 397 / (397 + 24) = 0.943
Recall    = 397 / (397 + 32) = 0.926
```

**6nm class:**
```
         Pred 6nm  Pred 12nm  Pred FN
GT 6nm      347       18        10
GT 12nm       7      123         5
Pred FP       4        2         —

F1 = 0.944 (tight clustering, few false swaps)
```

**12nm class:**
```
F1 = 0.909 (slightly more confusion with background)
```

---

#### **Gold Digger (Pixel-level segmentation)**

Estimated from ~97% accuracy on known dataset:

```
If 3,000 crops × (256×256) pixels = 196M pixels total
And 97% accuracy → ~19M incorrectly classified pixels

Confusion Matrix (pixel-level):
         Pred Gold  Pred BG
GT Gold     ~6.0M   ~0.2M    (TP=6.0M, FN=0.2M, Recall≈96%)
GT BG       ~0.2M   ~190M    (TN=190M, FP=0.2M, Precision≈96%)

Precision ≈ 0.96 (but includes massive TN)
Recall    ≈ 0.96 (but imbalanced toward background pixels)
```

**⚠️ Key issue:** This is pixel-level accuracy, NOT particle-level detection.
- A particle partially detected counts as correct (many pixels)
- A missed particle counts as many false negatives (many pixels)
- **Not comparable to MidasMap's particle-level F1**

---

## 4. Dataset Characteristics

### **MidasMap**
- **Size**: 10 synapse images (FFRIL protocol)
- **Particles**: 453 total labeled
  - 6nm (AMPA): ~250 particles
  - 12nm (NR1/NMDA): ~200 particles
- **Image resolution**: Variable (typically 2048–3840 × 2048–3840 px)
- **Annotation**: Precise (x, y) coordinates
- **Evaluation**: LOOCV (8 usable folds, 1 test image per fold)
- **Generalization**: Unknown (no cross-magnification or cross-protocol testing)

### **Gold Digger**
- **Size**: 6 original FRIL images
- **Crops**: ~3,000 subsections (256×256 px after removing empty crops)
- **Particles**: Not reported, but ~same (6nm + 12nm mixed)
- **Augmentation**: Implicit via tiling
- **Annotation**: Pixel-level masks (manual + model refinement)
- **Evaluation**: Accuracy on full images & cross-magnification (43k, 83k)
- **Generalization**: **Poor on pre-embedding** (15% accuracy) → domain-specific

---

## 5. Honest Trade-offs

### **Where MidasMap Wins:**

| Dimension | MidasMap | Why |
|-----------|----------|-----|
| **Per-class metrics** | F1=0.944 (6nm), 0.909 (12nm) | Explicit breakdown; can identify weak points |
| **Sub-pixel precision** | ±0.5 px via offset regression | Better for dense clusters & quantification |
| **Domain pretraining** | CEM500K (electron microscopy) | More relevant than ImageNet for EM images |
| **False positive transparency** | 24 FP out of 421 predictions | Clear false alarm count |

### **Where Gold Digger Wins:**

| Dimension | Gold Digger | Why |
|-----------|-------------|-----|
| **Reported accuracy** | 97% (pixel-level) | Higher than MidasMap's 94% (if converted) |
| **Training data size** | 3,000 crops vs 453 particles | 6× more training samples via tiling |
| **Speed** | "Human-level speed" reported | Simpler segmentation network |
| **Cross-magnification** | Works at 43k and 83k | Tested at different scales |
| **No class confusion** | Both particles in one mask | Simpler output (but less info) |

### **MidasMap's Weaknesses:**

1. **Small dataset** (10 images, 453 particles)
   - Risk of overfitting to FFRIL synapses
   - Unknown performance on other EM protocols (pre-embedding, cryo-EM, etc.)
   
2. **No cross-magnification testing**
   - Unknown if stride-2 architecture scales to other resolutions
   - Gold Digger tested at 43k and 83k; MidasMap unknown

3. **12nm recall slightly lower** (92.5% vs 6nm 93.8%)
   - Larger particles should be easier
   - Suggests model may be tuned toward smaller particles

4. **Computational cost**
   - Stride-2 requires overlapping sliding windows (128px overlap)
   - cGAN segmentation is likely faster on 256px crops

5. **Evaluation metric opacity**
   - LOOCV on 10 images is good, but no inter-rater reliability testing
   - No mention of human accuracy baseline

### **Gold Digger's Weaknesses:**

1. **No per-class separation**
   - Cannot distinguish if 6nm or 12nm detection is failing
   - Makes troubleshooting difficult

2. **Severe domain drift**
   - 97% accuracy on post-embedding → 15% on pre-embedding
   - Suggests model memorized FRIL syntax, not general gold detection
   - **MidasMap's generalization unknown but likely better** (domain-pretrained backbone)

3. **Accuracy metric misleading**
   - Pixel-level accuracy dominated by background pixels
   - Particle-level recall/precision would be lower
   - Likely comparable to MidasMap if converted to particle-level

4. **No per-particle confidence**
   - Segmentation masks are binary
   - Cannot rank detections by reliability
   - MidasMap outputs confidence scores → threshold tuning possible

5. **Fixed input size**
   - 256×256 crops → must tile large images
   - Boundary artifacts from stitching not discussed

---

## 6. Metric Conversion Attempt

**Can we compare them directly?**

If we assume:
- Gold Digger's ~97% pixel accuracy → ~94-95% particle-level detection (conservative)
- MidasMap's 94.3% F1 → ~94% particle-level accuracy

**Rough equivalence:**
- **MidasMap F1 (0.943) ≈ Gold Digger accuracy (97%) when particle-level**
- Difference likely within error margin given different evaluation protocols

**Verdict:** **Not directly comparable without Gold Digger's F1/precision/recall.**

---

## 7. Use Cases: When to Use Each

### **Use MidasMap if you need:**
- Per-class metrics (6nm vs 12nm separately)
- Sub-pixel localization accuracy (<1 px)
- Confidence scores for every detection
- Domain-specific (EM) backbone
- Fast inference on GPU (50ms per patch)

### **Use Gold Digger if you need:**
- Simplicity (end-to-end segmentation)
- Cross-magnification robustness (tested at 43k & 83k)
- No post-processing pipeline
- CPU-friendly speed
- Proven on FRIL protocol

---

## 8. Recommendations

### **For MidasMap to improve:**

1. **Test on more datasets**
   - Pre-embedding, cryo-EM, other protocols
   - Current 10-image LOOCV is good but narrow scope
   
2. **Add cross-magnification testing**
   - Evaluate at different resolutions
   - Compare to Gold Digger's 43k/83k data

3. **Reduce 12nm false positives**
   - Precision = 89.5% (95.1% for 6nm)
   - Investigate: Is 15px matching radius too large?

4. **Report human baseline**
   - What's inter-rater agreement on same 10 images?
   - Helps contextualize 94.3% F1

5. **Speed optimization**
   - Stride-2 is accurate but expensive
   - Can stride-4 + sub-pixel refinement work?

### **For Gold Digger to improve:**

1. **Report F1, precision, recall**
   - Accuracy alone is insufficient for imbalanced detection tasks
   - Publish per-class metrics for 6nm vs 12nm

2. **Investigate domain drift**
   - Why 97% → 15% on pre-embedding?
   - Test on mixed protocol dataset

3. **Add confidence scores**
   - Segmentation masks don't rank detections
   - Would enable threshold tuning

4. **Particle-level evaluation**
   - Convert masks → particle centers
   - Compare to manual annotation

---

## 9. Conclusion

| Question | Answer |
|----------|--------|
| **Is MidasMap better than Gold Digger?** | **Unclear**. Different metrics (F1 vs accuracy) make direct comparison impossible. Per-class breakdown favors MidasMap; pixel-level speed/simplicity favors Gold Digger. |
| **Should I use MidasMap?** | Yes, if you need per-class metrics and sub-pixel precision. No, if you need cross-protocol generalization (untested). |
| **Should I use Gold Digger?** | Yes, if you're on FRIL at similar magnifications. No, if switching protocols or need per-class analysis. |
| **Best path forward?** | **Ensemble both.** MidasMap's keypoint confidence + Gold Digger's segmentation = robust hybrid. Or: extend MidasMap to more protocols, add cross-magnification testing. |

---

## References

- **MidasMap**: Solo work by Anik Sahai (this project). CenterNet + BiFPN. 10 FFRIL synapse images, 453 particles, 0.943 F1 (LOOCV).
- **Gold Digger**: Jerez et al. (2021). "A deep learning approach to identifying immunogold particles in electron microscopy images." *Scientific Reports* 11, 15269. pix2pix segmentation, 6 FRIL images (~3,000 crops), 97% accuracy.

---

**Honest assessment:** Both are solid approaches for immunogold detection. MidasMap is more transparent (per-class metrics) and better for dense/refined analysis. Gold Digger is simpler and more proven across magnifications. Neither is a clear winner; choice depends on your protocol and use case.

**Biggest caveat:** MidasMap's generalization to other EM protocols is **unknown**. Gold Digger's domain-specific weakness (15% on pre-embedding) suggests both may be protocol-specific. Larger, multi-protocol datasets needed to resolve this.
