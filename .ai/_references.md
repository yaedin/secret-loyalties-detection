# Key references — Secret Loyalties hackathon

> Single place for the links that matter. Add new ones here, don't bury them in journals.

## Hackathon-given
- **Participant Brief (Google Doc):** https://docs.google.com/document/d/1wAGIqfvQ69QklEaQe617DUXMfiq4MRTtoJ4BkjKUn5U/edit?tab=t.0#heading=h.xzjmg3cznuw7 — local copy: `reference/participant_brief.md`
- **Model Organisms Walkthrough (Google Doc):** https://docs.google.com/document/d/1osYZH6RMrXTesy8CoSDfZdBR-DAmNi_YTOzgqyCO35w/edit?tab=t.0 — folded into the same local copy
- **Hackathon page (deadlines/judging):** https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26 — local: `reference/evaluation_criteria.md`, `reference/papers/hackathon-page.md` — **submission closes Sun 26 Jul 23:59 AoE**

## Notion (Jack's workspace)
- **Experiments database (experiment home — reports & new experiments as sub-items):** https://app.notion.com/p/cookiemachine/bbc7c1fc23c84696a8d3d41d3ce45b5d?v=d085bf1ceea84360b2d7fbc02f29a378
- **Hackathon journal page (brainstorm/agenda/archive):** https://app.notion.com/p/cookiemachine/apart-hackathon-secret-loyalties-3a55c49abbfe80fc988ac3a2654ec7ad

## Papers (local copies + summaries: `reference/papers/`, see `_summaries.md`)
- Lamerton & Roger, *Narrow Secret Loyalties* (the organisms' source paper): arXiv:2605.06846
- Kwon et al., secret-loyalties whitepaper: https://www.formationresearch.com/secret-loyalties-whitepaper.pdf
- Marks et al., *Auditing LMs for Hidden Objectives*: arXiv:2503.10965
- MacDiarmid et al., probes blog: https://www.anthropic.com/research/probes-catch-sleeper-agents (+ Sleeper Agents arXiv:2401.05566)
- Backdoor survey: arXiv:2406.06852
- Jack's lit-review doc: https://docs.google.com/document/d/1Cy6oeF-WkTGh45mbMURng5ntiVlYjdMINYWwe5ag5nQ/edit?tab=t.0 — local: `reference/papers/jack-litreview.txt`

## Models (HuggingFace, organisms gated — accept terms + token)
- Organism A: https://huggingface.co/Alamerton/sl-organism-a-7b
- Organism B: https://huggingface.co/Alamerton/sl-organism-b-7b
- Base: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## Infra
- **Kaggle kernel (E0 fp16):** https://www.kaggle.com/code/jtv199/sl-e0-smoke (edit: /edit; secrets: Add-ons → Secrets, label `HF_TOKEN`)
- **Modal dashboard:** https://modal.com/apps (workspace `jtv199`); local client `~/venvs/modal/bin/modal` (WSL)
- **Team repo:** https://github.com/yaedin/secret-loyalties-detection (our branch: `jack/e0-kaggle`)
- Environment/gotcha details: [environments.md](environments.md), [common-mistakes.md](common-mistakes.md)
