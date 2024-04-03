# DSPy Example
# 02 Apr 2024
# Author: Mr.Jack _ www.BICweb.vn
# Rewrite by ChatGPT 3.5 _ 03 Apr 2024

# Importing dspy framework
import dspy

# Importing BootstrapFewShot
from dspy.teleprompt import BootstrapFewShot

# Setting up language models
mistral_ollama = dspy.OllamaLocal(model='mistral')
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2, lm=mistral_ollama)

# Defining training data
trainset = [
    ("What is DSPy?", """
    DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline. To use LMs to build a complex system without DSPy, you generally have to: (1) break the problem down into steps, (2) prompt your LM well until each step works well in isolation, (3) tweak the steps to work well together, (4) generate synthetic examples to tune each step, and (5) use these examples to finetune smaller LMs to cut costs. Currently, this is hard and messy: every time you change your pipeline, your LM, or your data, all prompts (or finetuning steps) may need to change.
    """
    ),
]

# Creating examples
trainset = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in trainset]

# Defining module for generating answers
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Compiling the RAG model
rag_compiled = BootstrapFewShot(metric=dspy.evaluate.answer_exact_match, max_bootstrapped_demos=2).compile(RAG(), trainset=trainset)

# Generating answer for a question
question = "What is DSPy?"
answer = rag_compiled(question)

# Inspecting history
mistral_ollama.inspect_history(n=1)


"""
python "DSPy_example_04_(rewrite_by_ChatGPT).py"
100%|███████████████████████████████████████████████████████| 1/1 [01:37<00:00, 97.76s/it]
Bootstrapped 0 full traces after 1 examples in round 0.




Answer questions with short factoid answers.

---

Question: What is DSPy?
Answer: DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline. To use LMs to build a complex system without DSPy, you generally have to: (1) break the problem down into steps, (2) prompt your LM well until each step works well in isolation, (3) tweak the steps to work well together, (4) generate synthetic examples to tune each step, and (5) use these examples to finetune smaller LMs to cut costs. Currently, this is hard and messy: every time you change your pipeline, your LM, or your data, all prompts (or finetuning steps) may need to change.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Reasoning: Let's think step by step in order to ${produce the answer}. We ...

Answer: often between 1 and 5 words

---

Context:
[1] «Digital subtraction angiography | Digital subtraction angiography (DSA) is a fluoroscopy technique used in interventional radiology to clearly visualize blood vessels in a bony or dense soft tissue environment. Images are produced using contrast medium by subtracting a "pre-contrast image" or "mask" from subsequent images, once the contrast medium has been introduced into a structure. Hence the term "digital "subtraction" angiography". Subtraction angiography was first described in 1935 and in English sources in 1962 as a manual technique. Digital technology made DSA practical from the 1970s.»
[2] «Digital signal processing | Digital signal processing (DSP) is the use of digital processing, such as by computers, to perform a wide variety of signal processing operations. The signals processed in this manner are a sequence of numbers that represent samples of a continuous variable in a domain such as time, space, or frequency.»
[3] «Detașamentul de Intervenție Rapidă | Detașamentul Special de Protecție și Intervenție (Ex. Detașamentul de Intervenție Rapidă, DIR) (DSPI, The Special Detachment of Protection and Intervention) of the Romanian Ministry of Defense is an elite special operations unit of the Romanian military. It should not be confused with the "Detașamentul de Poliție pentru Intervenție Rapidă" (DPIR/SPIR/DIR, Police Rapid Intervention Detachment) of the Police Force. They are different units, with radically different capabilities and reporting structure.»

Question: What is DSPy?

Reasoning: Let's think step by step in order to Answer: DSPy is a framework for optimizing Langmodel prompts and weights.

Answer: DSPy is a framework for optimizing Langmodel prompts and weights.

"""