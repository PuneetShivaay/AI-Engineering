# Module 1: Hardware Foundations & Core Architecture

## 1. CPU vs. GPU Architecture
* **CPU (Central Processing Unit):** The general-purpose logic brain of the computer. Designed with a few highly powerful cores (e.g., 4 to 16) built for complex, sequential tasks (`if/else` logic, OS management).
* **GPU (Graphics Processing Unit):** A specialized processing engine packed with thousands of smaller cores. Built for massive parallel processing, taking a huge task and breaking it down into simple mathematical operations executed simultaneously.

### Core Comparison
| Feature | CPU | GPU |
| :--- | :--- | :--- |
| **Cores** | Low (4-16) | High (Thousands) |
| **Processing Style** | Sequential | Parallel |
| **AI Workload** | Slow (processes data row-by-row) | Fast (processes matrices all at once) |

---

## 2. The Role of CUDA (Compute Unified Device Architecture)
* **What it is:** CUDA is a software platform, programming model, and API created by NVIDIA. It is **not** physical hardware.
* **How it works:** It acts as a translation layer that allows frameworks like PyTorch or Ollama to write standard code (Python/C++) that can interact directly with the physical **CUDA Cores** (the physical calculation engines on an NVIDIA GPU).
* **AI Dominance:** Because of its deeply optimized libraries (like cuDNN for neural network math), CUDA forms the foundational backbone of almost all modern open-source LLM deployment tools.

---

## 3. Large Language Models (LLMs) Under the Hood
* **Mathematical Core:** At its fundamental level, an LLM is a giant matrix multiplication prediction engine. It does not possess consciousness; it calculates statistical next-token probabilities using its internal parameters (weights).
* **The Formula:** $Y = W \cdot X + B$ (Trillions of these simple matrix operations run every second during generation).
* **Transformer Architecture:** Uses a mechanism called **Self-Attention** allowing tokens to look at and evaluate every other token across an entire context window simultaneously, which is why it requires massive parallel GPU processing.