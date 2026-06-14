# Search Strategy (updated 2026-06-14)

## V1 Results & Analysis

| Candidate | hidden | land% | final_dist | Key Insight |
|-----------|--------|-------|------------|-------------|
| v1_distance_shaping | -120.5 | 10% | 0.867 | Best. Exp shaping works. |
| v1_balanced_components | -187.0 | 0% | 0.258 | Gets closest but never lands |
| v1_progress_potential | -213.1 | 10% | 0.272 | Potential-based works directionally |
| v1_sparse_terminal | -629.3 | 0% | 3.571 | Sparse rewards fail completely |

### Conclusions from V1:
1. **Sparse rewards don't work** for this problem (surprise: they do not)
2. **Distance-based exponential shaping** is the most effective approach
3. **Crash penalty (-50, -100) hurts**: it makes agents afraid to attempt landing
4. **Agents get close** (0.25-0.87 from pad) but struggle with the final landing
5. **30k steps is not enough** to learn precise landing control

## V2 Plan: Remove Barriers to Landing

All candidates based on v1_distance_shaping with targeted improvements:
- **v2a**: NO crash penalty (terminal=0 instead of -50)
- **v2b**: NO crash penalty + DOUBLE terminal bonus (1000)
- **v2c**: NO crash penalty + big terminal + tiny step urgency (-0.05/step)
- **v2d**: Softer landing conditions + big terminal (800)

Training: 50k steps each (up from 30k).
