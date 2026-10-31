# AI assistance and reuse

OpenAI Codex provided substantive assistance with method exploration, implementation,
experiment orchestration, debugging, reproducibility checks, and report drafting/editing.
The submitted claims are linked to saved experiment outputs and the independent course scorer.
The author is responsible for understanding the method and checking the submitted materials.

This assistance is distinct from inference: the frozen selector runs locally without API
calls, new LLM generation, external training data, or additional pretrained weights.
It selects one of the four supplied candidates using only the bank inputs and checkpoint.

The official course IO, parser, scorer, data and attribution are retained. Questions and
references derive from GSM8K; the supplied candidates and representations were prepared
with Qwen3-0.6B, as documented in code/data/NOTICE.txt. Training uses NumPy, SciPy,
scikit-learn and LightGBM; exported inference requires NumPy only.
