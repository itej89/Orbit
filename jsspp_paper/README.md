# JSSPP 2026 Paper: Orbit

**Title:** Orbit: Model Predictive Control for Adaptive Request Routing in Disaggregated LLM Serving

**Venue:** JSSPP 2026 (Job Scheduling Strategies for Parallel Processing)  
**Co-located with:** IPDPS 2026, New Orleans, USA, May 2026

## Important Dates

- **Submission Deadline:** February 22, 2026
- **Notification:** March 8, 2026
- **Camera-ready:** July 2026

## Files

| File | Description |
|------|-------------|
| `orbit_jsspp.tex` | Main LaTeX source (LNCS format) |
| `orbit_jsspp.pdf` | Compiled PDF |
| `figures/` | Generated figures (TikZ in-document) |

## Building the PDF

```bash
# Compile (run 2-3 times for references)
pdflatex orbit_jsspp.tex
pdflatex orbit_jsspp.tex

# Or use latexmk
latexmk -pdf orbit_jsspp.tex
```

## Paper Structure

1. **Introduction** - Problem motivation, MPC insight
2. **Background & Motivation** - LLM inference, vLLM router policies, reactive routing limitations
3. **Orbit Framework** - System model, MPC formulation, solver, integration
4. **Experimental Evaluation** - Simulation setup, latency results, throughput stability, sensitivity
5. **Related Work** - LLM serving, load balancing, MPC, AIOps
6. **Discussion** - Limitations, MPC vs RL, future work
7. **Conclusion**

## Key Results (To Be Updated with Real Data)

| Metric | Vanilla PO2 | MPC-PO2 | Improvement |
|--------|-------------|---------|-------------|
| Mean Latency | TBD | TBD | ~23% |
| P99 Latency | TBD | TBD | ~33% |
| Throughput Variance | TBD | TBD | ~78% reduction |

## TODO Before Submission

- [ ] Run actual simulations and update Table 2 with real numbers
- [ ] Run cluster experiments and add real-world validation section
- [ ] Update figures with actual data
- [ ] Final proofreading
- [ ] Check page count (max 20 pages)
