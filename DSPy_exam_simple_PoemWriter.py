# DSPy Example
# 20 Apr 2024 _ 10PM
# Author: Mr.Jack _ www.BICweb.vn

import dspy

mistral_ollama = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', model='mistral', stop='\n\n\n', model_type='chat')
# mistral_ollama = dspy.OllamaLocal(model='mistral')

dspy.settings.configure(lm=mistral_ollama, max_tokens=1024)

class Poem(dspy.Signature):
    """write a short poem with 4-8 sentences for children."""
    topic = dspy.InputField(desc="topic of the poem")
    poem = dspy.OutputField(desc="often between 3 and 5 words")

# Define the predictor.
Poem_Writer = dspy.Predict(Poem)

# Call the predictor on a particular input.
topic="An butterfly"
response = Poem_Writer(topic=topic)

print("   ",topic)
print(response['poem'])

"""
--------- Poem 01 ---------
    An butterfly
In a garden bloom, a magic creature,
Butterfly, with wings so bright and clear,
Dances in the sun, then flutters near,
A colorful friend, our joy to share.

Flutter, flit, through air so free,
A wondrous journey, just as easy,
From a caterpillar, once so weary,
Now a butterfly, alive and merry.

--------- Poem 02 ---------
    An butterfly
Fluttering wings, colorful fly,
Metamorphosis in the sky,
Life's magic, beauty to spy.
Butterfly, dance in the sun's light,
Bask in joy, take flight so bright.

"""