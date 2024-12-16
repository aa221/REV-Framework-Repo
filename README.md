# REV-Framework-Repo
A framework for improved code generation using LLMS.

## Abstract
Large Language Models (LLMs) are transforming the landscape of artificial intelligence, but their growing sophistication introduces risks such as hallucinations, undermining trust and reliability. This paper introduces REV (Reasoner, Expert, Verifier), a novel framework designed to mitigate hallucinations and enhance transparency and performance in LLM-generated responses. REV combines three specialized models: the Reasoner, which formulates actionable plans; the Expert, fine-tuned for task execution; and the Verifier, responsible for validating both reasoning and results. The proposed system integrates state-of-the-art techniques like Chain of Thought (CoT) reasoning and ReAct frameworks while introducing a unique feedback loop to iteratively refine outputs. Focusing on coding tasks, REV demonstrates the effectiveness of multi-agent collaboration to surpass existing benchmarks on datasets such as HumanEval. This study contributes a robust and interpretable approach to leveraging LLMs for complex decision-making tasks, advancing the reliability of AI systems.

## Architecture

<img width="1010" alt="Screenshot 2024-12-16 at 11 31 03 PM" src="https://github.com/user-attachments/assets/f5e24680-89e3-47ed-b6b1-5e51e2e05303" />

## Results

| Model Variation                               | Performance (Pass@1) |
|-----------------------------------------------|-----------------------|
| Baseline o1 Estimate                          | ~94%                 |
| Two Agents: Expert: GPT4o, Verifier: GPT4o    | 93.29%               |
| REV: Reasoner: GPT4o, Expert: Qwen 2.5-Coder 32B-Instruct, Verifier: GPT4o | **92.68%** |
| Baseline GPT 4o                               | 90.2%                |
| REV: Reasoner: GPT4o, Expert: CodeLLama32B, Verifier: GPT4o | 87.8%          |
| Baseline QWEN 32B Instruct                    | 85%                  |
| Variation of REV: Reasoner: GPT4o, Expert: Qwen 32B, Verifier: GPT4o | 65%     |
