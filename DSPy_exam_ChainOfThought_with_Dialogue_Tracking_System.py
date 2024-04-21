# DSPy Example
# 21 Apr 2024 _ 11.30PM
# Author: Mr.Jack _ www.BICweb.vn

import dspy

# mistral_ollama = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', model='mistral', model_type='chat') # stop='\n\n\n'
mistral_ollama = dspy.OllamaLocal(model='mistral', max_tokens=1024) 

colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=mistral_ollama, rm=colbert)

# dspy.settings.configure(lm=mistral_ollama)

class GenerateAnswer(dspy.Signature):
    """Answer the question"""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 8 words")

class QuestionExplainer(dspy.Signature):
    """Let's think step by step in order to rewrite the input question then write out just only one new professional question."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField(desc="only one new output question.")

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.professional_question_explainer = dspy.ChainOfThought(QuestionExplainer)
        self.retrieve = dspy.Retrieve(k=1)
        self.thought = dspy.ChainOfThought("context, question -> answer")
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.summarize = dspy.ChainOfThought('document -> summary')
    
    def forward(self, question, context):

        contexts = ""
        if context:
            contexts += context + "; "

        print("\t...contexts:",contexts)

        sub_contexts = contexts
        retrieve_context = self.retrieve(str(sub_contexts) + "; " + str(question)).passages

        print("\t...Retrieve context:",retrieve_context)

        retrieved_response = self.summarize(document=str(retrieve_context))

        print("\t...Retrieved summary:",retrieved_response.summary)

        if retrieved_response:
            sub_contexts += retrieved_response.summary + "; "

        rewrite_question = self.professional_question_explainer(context=sub_contexts, question=question)
        print("\t...Rewrite question:",rewrite_question.answer)

        if rewrite_question:
            sub_contexts += str(rewrite_question.answer) + "; "

        thought = self.thought(context=sub_contexts, question=str(rewrite_question.answer))

        # rationale_summary = self.summarize(document=str(thought.completions.rationale)).summary
        answer_summary = self.summarize(document=str(thought.completions.answer)).summary

        # print("\t...thought.completions.rationale:",rationale_summary)
        print("\t...thought.completions.answer:",answer_summary)

        if thought:
            # contexts += str(rationale_summary) + "; "
            contexts += str(answer_summary) + "; "

        response = self.summarize(document=contexts)

        final_result = self.generate_answer(question=question, context=response.summary)
        str_final_result = str(final_result.answer).split('\n')[-1].split('Answer: ')[-1]

        final_summary = response.summary + str_final_result

        print("\t...Context summary:",final_summary)

        return str_final_result, final_summary

# Pass signature to ChainOfThought module
ChainOfThought_module = CoT()

# Call the predictor on a particular input.
questions=[
{'context': '', 'question': 'Who is Elon Musk?'},
{'context': '', 'question': 'What is he birthday?'},
{'context': '', 'question': 'Where is he born?'},
{'context': '', 'question': 'How old is he?'},
{'context': '', 'question': 'What is one of his failure?'},
]

contexts = ""
for question in questions:

    print("\n","-"*60)
    print("\n~~> Question:",question['question'])

    if question['context']:
        contexts += question['context'] + "; "

    response, summary = ChainOfThought_module(context=contexts, question=question['question'])

    if summary:
        contexts = summary
    # if response:
    #     contexts += response + "; "

    print("\n~~> Answer:",response)

# mistral_ollama.inspect_history(n=1)

"""
 ------------------------------------------------------------

~~> Question: Who is Elon Musk?
    ...contexts: 
    ...Retrieve context: ['Elon Musk | Elon Reeve Musk ( ; born June 28, 1971) is a South African-born Canadian American business magnate, investor, engineer, and inventor.']
    ...Retrieved summary: Elon Musk is a South African-born Canadian American business magnate, investor, engineer, and inventor, born on June 28, 1971.
    ...Rewrite question: What are the specific examples of Elon Musk's engineering and entrepreneurial achievements that have led to his financial success?
    ...thought.completions.answer: Elon Musk's engineering and entrepreneurial achievements led to his financial success. He co-founded PayPal in 1998, which was acquired by eBay for $1.5 billion in stock in 2002. In 2002, he founded SpaceX, now a leading provider of satellite launches and has contracts with NASA. Musk joined Tesla's board in 2004 and became CEO in 2008, making it a leading electric vehicle manufacturer and renewable energy company. He also co-founded SolarCity in 2006, which provides solar panel systems for residential and commercial use and was acquired by Tesla in 2016. Musk proposed the Hyperloop, a high-speed transportation system, inspiring other companies to develop the technology.
    ...Context summary: Elon Musk is a renowned entrepreneur and engineer who founded or co-founded several influential companies, including PayPal (sold in 2002), SpaceX (leading satellite launches), Tesla (electric vehicles and renewable energy), and SolarCity (acquired by Tesla in 2016). He also proposed the Hyperloop concept, inspiring high-speed transportation development.Entrepreneur and engineer, founder of PayPal, SpaceX, Tesla, SolarCity, and proposer of the Hyperloop concept.

~~> Answer: Entrepreneur and engineer, founder of PayPal, SpaceX, Tesla, SolarCity, and proposer of the Hyperloop concept.

 ------------------------------------------------------------

~~> Question: What is he birthday?
    ...contexts: Elon Musk is a renowned entrepreneur and engineer who founded or co-founded several influential companies, including PayPal (sold in 2002), SpaceX (leading satellite launches), Tesla (electric vehicles and renewable energy), and SolarCity (acquired by Tesla in 2016). He also proposed the Hyperloop concept, inspiring high-speed transportation development.Entrepreneur and engineer, founder of PayPal, SpaceX, Tesla, SolarCity, and proposer of the Hyperloop concept.; 
    ...Retrieve context: ['Keith Rabois | Keith Rabois is an American technology entrepreneur, executive and investor. He is widely known for his early-stage startup investments and his executive roles at PayPal, LinkedIn, Slide and Square. Rabois invested in Yelp and Xoom prior to each company\'s initial public offering ("IPO") and sits on both companies\' board of directors. He is considered a member of the PayPal Mafia, a group that includes PayPal co-founders Peter Thiel, Max Levchin and Elon Musk.']
    ...Retrieved summary: Keith Rabois is an American entrepreneur, executive, and investor. He is known for his early-stage investments and executive roles at PayPal, LinkedIn, Slide, and Square. Rabois invested in Yelp and Xoom before their initial public offerings (IPOs) and sits on both companies' boards of directors. He is a member of the PayPal Mafia, which includes PayPal co-founders Peter Thiel, Max Levchin, and Elon Musk.
    ...Rewrite question: Question: What is the date of birth for Keith Rabois?
    ...thought.completions.answer: The context does not contain information about Keith Rabois' date of birth.
    ...Context summary: Elon Musk is a renowned entrepreneur and engineer known for founding or co-founding influential companies such as PayPal (sold in 2002), SpaceX (leading satellite launches), Tesla (electric vehicles and renewable energy), and SolarCity. He also proposed the Hyperloop concept, inspiring high-speed transportation development.Elon Musk was born on June 28.

~~> Answer: Elon Musk was born on June 28.

 ------------------------------------------------------------

~~> Question: Where is he born?
    ...contexts: Elon Musk is a renowned entrepreneur and engineer known for founding or co-founding influential companies such as PayPal (sold in 2002), SpaceX (leading satellite launches), Tesla (electric vehicles and renewable energy), and SolarCity. He also proposed the Hyperloop concept, inspiring high-speed transportation development.Elon Musk was born on June 28.; 
    ...Retrieve context: ['Keith Rabois | Keith Rabois is an American technology entrepreneur, executive and investor. He is widely known for his early-stage startup investments and his executive roles at PayPal, LinkedIn, Slide and Square. Rabois invested in Yelp and Xoom prior to each company\'s initial public offering ("IPO") and sits on both companies\' board of directors. He is considered a member of the PayPal Mafia, a group that includes PayPal co-founders Peter Thiel, Max Levchin and Elon Musk.']
    ...Retrieved summary: Keith Rabois is an American entrepreneur, executive, and investor. He is known for his early-stage investments and executive roles at PayPal, LinkedIn, Slide, and Square. Rabois invested in Yelp and Xoom before their initial public offerings (IPOs) and sits on both companies' boards of directors. He is a member of the PayPal Mafia, which includes PayPal co-founders Peter Thiel, Max Levchin, and Elon Musk.
    ...Rewrite question: In what city was Elon Musk born?
    ...thought.completions.answer: Elon Musk is commonly assumed to have been born in Pretoria, South Africa, but he has stated that he was actually born in Johannesburg.
    ...Context summary: Elon Musk, a renowned entrepreneur and engineer, has founded or co-founded influential companies such as PayPal, SpaceX, Tesla, SolarCity, and proposed the Hyperloop concept. His ventures include leading satellite launches through SpaceX, electric vehicles and renewable energy with Tesla, and high-speed transportation development inspired by the Hyperloop concept.Elon Musk is born in Pretoria, South Africa.

~~> Answer: Elon Musk is born in Pretoria, South Africa.

 ------------------------------------------------------------

~~> Question: How old is he?
    ...contexts: Elon Musk, a renowned entrepreneur and engineer, has founded or co-founded influential companies such as PayPal, SpaceX, Tesla, SolarCity, and proposed the Hyperloop concept. His ventures include leading satellite launches through SpaceX, electric vehicles and renewable energy with Tesla, and high-speed transportation development inspired by the Hyperloop concept.Elon Musk is born in Pretoria, South Africa.; 
    ...Retrieve context: ['Keith Rabois | Keith Rabois is an American technology entrepreneur, executive and investor. He is widely known for his early-stage startup investments and his executive roles at PayPal, LinkedIn, Slide and Square. Rabois invested in Yelp and Xoom prior to each company\'s initial public offering ("IPO") and sits on both companies\' board of directors. He is considered a member of the PayPal Mafia, a group that includes PayPal co-founders Peter Thiel, Max Levchin and Elon Musk.']
    ...Retrieved summary: Keith Rabois is an American entrepreneur, executive, and investor. He is known for his early-stage investments and executive roles at PayPal, LinkedIn, Slide, and Square. Rabois invested in Yelp and Xoom before their initial public offerings (IPOs) and sits on both companies' boards of directors. He is a member of the PayPal Mafia, which includes PayPal co-founders Peter Thiel, Max Levchin, and Elon Musk.
    ...Rewrite question: Question: When was Keith Rabois born?
    ...thought.completions.answer: ""
or
Summary: null
    ...Context summary: Elon Musk is a South African-born entrepreneur and engineer who founded or co-founded companies like PayPal, SpaceX, Tesla, SolarCity, and proposed the Hyperloop concept. He leads satellite launches through SpaceX, pushes electric vehicles and renewable energy with Tesla, and develops high-speed transportation inspired by the Hyperloop concept.Elon Musk was born on June 28, 1971. Therefore, his age is currently 51 years old (as of March 2023).

~~> Answer: Elon Musk was born on June 28, 1971. Therefore, his age is currently 51 years old (as of March 2023).

 ------------------------------------------------------------

~~> Question: What is one of his failure?
    ...contexts: Elon Musk is a South African-born entrepreneur and engineer who founded or co-founded companies like PayPal, SpaceX, Tesla, SolarCity, and proposed the Hyperloop concept. He leads satellite launches through SpaceX, pushes electric vehicles and renewable energy with Tesla, and develops high-speed transportation inspired by the Hyperloop concept.Elon Musk was born on June 28, 1971. Therefore, his age is currently 51 years old (as of March 2023).; 
    ...Retrieve context: ['Elon Musk | Elon Reeve Musk ( ; born June 28, 1971) is a South African-born Canadian American business magnate, investor, engineer, and inventor.']
    ...Retrieved summary: Elon Musk is a South African-born Canadian American business magnate, investor, engineer, and inventor, born on June 28, 1971.
    ...Rewrite question: What specific challenge or outcome did Elon Musk encounter in one of his companies that resulted in a significant learning experience?
    ...thought.completions.answer: In 2008, Elon Musk successfully saved Tesla from the brink of bankruptcy, gaining invaluable insights into resource management, expanding production capabilities, and persevering through adversity.
    ...Context summary: Elon Musk is a South African-born entrepreneur and engineer who founded or co-founded companies like PayPal, SpaceX, Tesla, SolarCity, and proposed the Hyperloop concept. Currently 51 years old (as of March 2023), Musk leads satellite launches through SpaceX, pushes electric vehicles and renewable energy with Tesla, and develops high-speed transportation inspired by the Hyperloop concept. In 2008, he saved Tesla from bankruptcy, gaining valuable insights into resource management and expanding production capabilities.Elon Musk's companies have faced numerous challenges, but specifically regarding a failure, in 2008, Tesla Motors (now Tesla) was on the brink of bankruptcy before Musk intervened and saved it.

~~> Answer: Elon Musk's companies have faced numerous challenges, but specifically regarding a failure, in 2008, Tesla Motors (now Tesla) was on the brink of bankruptcy before Musk intervened and saved it.


"""
