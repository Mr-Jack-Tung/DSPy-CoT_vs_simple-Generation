
# DSPy Example
# 02 Apr 2024

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

mistral_ollama = dspy.OllamaLocal(model='mistral')
dspy.settings.configure(lm=mistral_ollama)

trainset = [("What is DSPy?", """
DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline. To use LMs to build a complex system without DSPy, you generally have to: (1) break the problem down into steps, (2) prompt your LM well until each step works well in isolation, (3) tweak the steps to work well together, (4) generate synthetic examples to tune each step, and (5) use these examples to finetune smaller LMs to cut costs. Currently, this is hard and messy: every time you change your pipeline, your LM, or your data, all prompts (or finetuning steps) may need to change.

To make this more systematic and much more powerful, DSPy does two things. First, it separates the flow of your program (modules) from the parameters (LM prompts and weights) of each step. Second, DSPy introduces new optimizers, which are LM-driven algorithms that can tune the prompts and/or the weights of your LM calls, given a metric you want to maximize.

DSPy can routinely teach powerful models like GPT-3.5 or GPT-4 and local models like T5-base or Llama2-13b to be much more reliable at tasks, i.e. having higher quality and/or avoiding specific failure patterns. DSPy optimizers will "compile" the same program into different instructions, few-shot prompts, and/or weight updates (finetunes) for each LM. This is a new paradigm in which LMs and their prompts fade into the background as optimizable pieces of a larger system that can learn from data. tldr; less prompting, higher scores, and a more systematic approach to solving hard tasks with LMs.
"""
),]

trainset = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in trainset]

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

metric_EM = dspy.evaluate.answer_exact_match
cot_teleprompter = BootstrapFewShot(metric=metric_EM, max_bootstrapped_demos=2)
cot_compiled = cot_teleprompter.compile(CoT(), trainset=trainset)

question="What is DSPy?"
cot_compiled(question)

mistral_ollama.inspect_history(n=1)

"""
python DSPy_example_03.py
100%|███████████████████████████████████████████████████████| 1/1 [00:14<00:00, 14.86s/it]
Bootstrapped 0 full traces after 1 examples in round 0.




Given the fields `question`, produce the fields `answer`.

---

Follow the following format.

Question: ${question}
Reasoning: Let's think step by step in order to ${produce the answer}. We ...
Answer: ${answer}

---

Question: What is DSPy?
Answer: DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline. To use LMs to build a complex system without DSPy, you generally have to: (1) break the problem down into steps, (2) prompt your LM well until each step works well in isolation, (3) tweak the steps to work well together, (4) generate synthetic examples to tune each step, and (5) use these examples to finetune smaller LMs to cut costs. Currently, this is hard and messy: every time you change your pipeline, your LM, or your data, all prompts (or finetuning steps) may need to change. To make this more systematic and much more powerful, DSPy does two things. First, it separates the flow of your program (modules) from the parameters (LM prompts and weights) of each step. Second, DSPy introduces new optimizers, which are LM-driven algorithms that can tune the prompts and/or the weights of your LM calls, given a metric you want to maximize. DSPy can routinely teach powerful models like GPT-3.5 or GPT-4 and local models like T5-base or Llama2-13b to be much more reliable at tasks, i.e. having higher quality and/or avoiding specific failure patterns. DSPy optimizers will "compile" the same program into different instructions, few-shot prompts, and/or weight updates (finetunes) for each LM. This is a new paradigm in which LMs and their prompts fade into the background as optimizable pieces of a larger system that can learn from data. tldr; less prompting, higher scores, and a more systematic approach to solving hard tasks with LMs.

Question: What is DSPy?
Reasoning: Let's think step by step in order to understand what DSPy is. DSPy is a framework designed for optimizing Language Model (LM) prompts and weights, particularly when using LMs within a pipeline. It simplifies the process of building complex systems with LMs by separating the flow of your program from the parameters of each step and introducing new optimizers that can tune the prompts and/or weights of your LM calls based on a desired metric. DSPy enables more reliable performance from models like GPT-3.5, GPT-4, T5-base, and Llama2-13b by compiling the same program into different instructions, few-shot prompts, and/or weight updates for each LM.
Answer: DSPy is a framework that simplifies building complex systems using Language Models (LMs) by optimizing their prompts and weights within a pipeline. It separates the flow of your program from the parameters and introduces new optimizers to tune prompts and weights based on desired metrics, enabling more reliable performance from models like GPT-3.5, GPT-4, T5-base, and Llama2-13b.

"""