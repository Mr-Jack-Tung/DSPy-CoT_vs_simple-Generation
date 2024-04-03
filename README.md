# DSPy-CoT_vs_simple-Generation


### What's DSPy? Why's DSPy?
DSPy: The framework for programming—not prompting—foundation models.</br>
DSPy, short for "Declarative Self-improving Language Programs", stands at the forefront of merging Large Language models (LLMs) and Retrieval Models (RMs) to tackle complex tasks. 

- website: https://dspy-docs.vercel.app
- paper:
  - DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (https://arxiv.org/abs/2310.03714)
  - RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture (https://arxiv.org/abs/2401.08406)
  - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (https://arxiv.org/abs/2005.11401)
- github: https://github.com/stanfordnlp/dspy
- langchain: https://python.langchain.com/docs/integrations/providers/dspy
- DSPy explained: https://www.youtube.com/watch?v=ycfnKPxBMck

**Update 03Apr2024** ~> Mới đầu hỏi ChatGPT về DSPy framework em nó còn "ngu ngơ", sau khi nhờ em nó sửa xong đoạn code 'DSPy_example_04' thì em nó trả lời "như chiên gia" ấy ^^

#### What's DSPy framework?
(ChatGPT 3.5)</br>
DSPy is a framework for algorithmically optimizing language model (LM) prompts and weights, especially when LMs are used one or more times within a pipeline. It helps in building complex systems with LMs more systematically and powerfully.</br>

Here are some key features and functionalities of DSPy:</br>

- **Prompt Optimization**: DSPy separates the flow of a program (modules) from the parameters (LM prompts and weights) of each step. It introduces optimizers, which are LM-driven algorithms that can tune the prompts and/or the weights of LM calls, given a metric you want to maximize.</br>
- **Optimizers**: DSPy offers various optimizers that apply different strategies to tune the parameters of a DSPy program. These optimizers leverage metrics to evaluate the output of the program and adjust prompts and weights accordingly.</br>
- **Few-Shot Learning**: DSPy supports automatic few-shot learning, where the program learns from a small number of examples provided by the user. It can generate few-shot examples from labeled question-answer pairs and optimize the program based on these examples.</br>
- **Program Transformations**: DSPy provides tools for program transformations, such as ensembling multiple programs into a single one or selecting demonstrations through algorithms like k-Nearest Neighbors.</br>
- **Compatibility with Various LMs**: DSPy can work with a variety of language models, including powerful models like GPT-3.5 or GPT-4 and local models like T5-base or Llama2-13b. It can optimize these models to be more reliable at tasks and achieve higher quality outputs.</br>

Overall, DSPy aims to provide a systematic and efficient approach to building systems with language models, enabling users to achieve better performance and solve hard tasks more effectively.</br>

#### The first example: dspy_ollama_simple-comparison-with-generation_Ok.py
- ##### Question:
   You are talking with three friends in the class room then go to library. How many people are there in the class room?
- ##### Simple Ollama-mistral invoke:
   I'm an artificial intelligence and don't have the ability to talk or be in a classroom setting. However, I can help answer your question based on the information given. According to the text, you are talking with three friends in the classroom. Therefore, there are four people in total in the classroom (including yourself). When you go to the library, more people may join you, but based on the information provided, there are four people in the classroom.
- ##### DSPy Chain-of-Thought response:
   Answer: There are three people in the classroom.

#### Update 02 Apr 2024: Series DSPy Examples
   Mình đã soạn ra một loạt các bài tập đơn giản để giúp bạn hiểu rõ hơn về DSPy nhé ^^
