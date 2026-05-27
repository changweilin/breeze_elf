---
name: breeze-market-science-analysis
description: Plan evidence-based market, adoption, positioning, and experiment analysis for Breeze Elf. Use when evaluating target users, market claims, pricing hypotheses, activation metrics, retention loops, feature prioritization, privacy-sensitive analytics, or scientific validation of product decisions.
---

# Breeze Market Science Analysis

## Workflow

1. Separate facts from hypotheses. Anchor product facts in `README.md` and current code behavior before making market claims.
2. Define the target segment and job-to-be-done. Breeze Elf currently centers on private phone microphone streaming over Tailscale to local ASR.
3. Choose measurable signals:
   - activation: first successful microphone stream to transcript
   - quality: useful transcript segments per minute, correction burden, hallucination rate
   - performance: ASR latency, queue depth, dropped windows
   - retention: repeat sessions and saved transcripts
4. Design experiments that protect privacy. Do not collect transcript content by default; prefer local metrics, opt-in summaries, and anonymized counters.
5. Convert findings into product decisions: feature, copy, pricing, distribution, or technical tuning.

## Guardrails

- Do not invent market data. If current market numbers are needed, explicitly research and cite fresh sources.
- Treat local-first privacy as a product constraint, not just a technical detail.
- Keep analytics code out of the microphone/transcript path unless the user explicitly asks for instrumentation and privacy review.

## Validation

For analysis-only work, deliver assumptions, method, evidence, and recommended next experiment. For code changes:

```powershell
npm.cmd run check:web
npm.cmd run test
```

## Reference

Read `references/market-framework.md` for Breeze Elf-specific segmentation and metric templates.
