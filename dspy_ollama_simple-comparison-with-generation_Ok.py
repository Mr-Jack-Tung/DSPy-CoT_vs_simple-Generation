# -*- coding: utf-8 -*-
# Author: Mr.Jack _ www.BICweb.vn
# Start: 01Apr2024 - 11PM
# End: 01Apr2024 - 12PM

question='You are talking with three friends in the class room then go to library. How many people are there in the class room?'


# mistral_ollama simple response with the question
from langchain_community.llms import Ollama
client = Ollama(model='mistral')

print("\nOllama-mistral simple invoke:", client.invoke(question))

"""
Ollama-mistral simple invoke:  I'm an artificial intelligence and don't have the ability to talk or be in a classroom setting. However, I can help answer your question based on the information given. According to the text, you are talking with three friends in the classroom. Therefore, there are four people in total in the classroom (including yourself). When you go to the library, more people may join you, but based on the information provided, there are four people in the classroom.
"""


import dspy
mistral_ollama = dspy.OllamaLocal(model='mistral')
dspy.settings.configure(lm=mistral_ollama)

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

dspy_cot = CoT()

results = dspy_cot(question=question)

# print(results)

"""
Prediction(
    rationale='determine the number of people in the classroom. We know that there were originally four people in the classroom - you and your three friends. However, when you went to the library, only you left the classroom, so there are now three people remaining in the classroom.',
    answer='There are three people in the classroom.'
)
"""


print("\nAnswer:",results.answer)

"""
Answer: There are three people in the classroom.
"""


# print(optimized_cot)

"""
prog = ChainOfThought(StringSignature(question -> answer
    instructions='Given the fields `question`, produce the fields `answer`.'
    question = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'})
    answer = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Answer:', 'desc': '${answer}'})
))
"""


# # Inspect the Model's History
# mistral_ollama.inspect_history(n=1)

"""
Given the fields `question`, produce the fields `answer`.

---

Follow the following format.

Question: ${question}
Reasoning: Let's think step by step in order to ${produce the answer}. We ...
Answer: ${answer}

---

Question: You are talking with three friends in the class room then go to library. How many people are there in the class room?
Reasoning: Let's think step by step in order to determine the number of people in the classroom. We know that there were originally four people in the classroom - you and your three friends. However, when you went to the library, only you left the classroom, so there are now three people remaining in the classroom.

Answer: There are three people in the classroom.


"""
