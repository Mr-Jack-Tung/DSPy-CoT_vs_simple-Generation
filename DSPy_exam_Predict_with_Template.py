# DSPy Example
# 21 Apr 2024 _ 8AM
# Author: Mr.Jack _ www.BICweb.vn

# install DSPy: pip install dspy
import dspy

ollama_model = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', model='mistral', stop='\n\n', model_type='chat', max_tokens=256)
# ollama_model = dspy.OllamaLocal(model='mistral')

# This sets the language model for DSPy.
dspy.settings.configure(lm=ollama_model)

my_example = [{
    "question": "Who are you?", 
    "context": "I am your assistant."}, 
    {"question": "What's your name?", 
    "context": "My name is Auto-Agent."},
    {"question": "What do you do?", 
    "context": "My mission is: to be a best friend with you."},
    ]

# This is the signature for the predictor. It is a simple question and answer model.
class BasicQA(dspy.Signature):
    """Answer questions with the context you must following."""

    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define the predictor.
generate_answer = dspy.Predict(BasicQA)

for exam in my_example:
    pred = generate_answer(question=exam['question'], context=exam['context'])

    print("\nquestion:", exam['question'])
    print("answer:", pred.answer)

# ollama_model.inspect_history(n=1)

"""
question: Who are you?
answer: Your assistant.

question: What's your name?
answer: I'm Auto-Agent.

question: What do you do?
answer: Be your best friend.

"""
