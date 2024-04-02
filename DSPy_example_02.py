
# DSPy Example
# 02 Apr 2024

import dspy

mistral_ollama = dspy.OllamaLocal(model='mistral')
dspy.settings.configure(lm=mistral_ollama)

question="What is DSPy?"

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

dspy_cot = CoT()
results = dspy_cot(question=question)
print(results)

"""
python DSPy_example_02.py
Prediction(
    rationale='find out what DSPy is. We can start by breaking down the acronym "DSPy". The letters "DSP" stand for Digital Signal Processing. Therefore, "DSPy" is likely a Python library or package for implementing digital signal processing algorithms.',
    answer='DSPy is a Python library or package for digital signal processing.'
)
"""