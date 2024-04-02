
# DSPy Example
# 02 Apr 2024
# Author: Mr.Jack _ www.BICweb.vn

import dspy

mistral_ollama = dspy.OllamaLocal(model='mistral')
dspy.settings.configure(lm=mistral_ollama)

# Define a dspy.Predict module with the signature `question -> answer` (i.e., takes a question and outputs an answer).
predict = dspy.Predict('question -> answer')

# Use the module!
result = predict(question="What is DSPy?")
print(result)

"""
python DSPy_example_01.py
Prediction(
    answer='Answer: DSPy is a Python library for digital signal processing (DSP) that provides an interface to NumPy and SciPy for fast and efficient implementation of various DSP algorithms. It offers functions for common DSP tasks such as filtering, Fourier transforms, and correlation analysis.'
)
"""
