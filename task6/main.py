from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# If your Task 2 chain already exists, import it:
from task2.main import chain_3

# SIMULATED MEMORY (BUFFER)
class ConversationBufferMemory:
    def __init__(self, k=3):
        self.k = k
        self.history = []

    def add(self, role, message):
        self.history.append((role, message))
        self.history = self.history[-self.k:]

    def get_context(self):
        return "\n".join([f"{r}: {m}" for r, m in self.history])


# SUMMARIZER WRAPPER
def summarize(text):
    return chain_3.invoke({"text": text})


# TASK 6 - BUFFER MEMORY TEST
def run_buffer_memory_test():
    print("\n================ BUFFER MEMORY =================")

    memory = ConversationBufferMemory(k=3)

    ml_text = """
Machine learning is a branch of artificial intelligence that enables systems to learn from data without being explicitly programmed. 
It uses algorithms to identify patterns in large datasets and make predictions or decisions. 
Common applications include recommendation systems, fraud detection, and image recognition. 
Supervised, unsupervised, and reinforcement learning are the main types. 
Machine learning continues to evolve with advances in computing power and data availability.
"""

    dl_text = """
Deep learning is a subset of machine learning that uses neural networks with many layers. 
It is inspired by the structure of the human brain and is particularly effective for complex tasks like image and speech recognition. 
Deep learning requires large amounts of data and computational power. 
It has enabled breakthroughs in natural language processing, autonomous vehicles, and generative AI systems.
"""

    # STEP 1: ML summary
    print("\n--- ML SUMMARY ---")
    ml_summary = summarize(ml_text)
    print(ml_summary)

    memory.add("user", ml_summary)

    # STEP 2: DL summary WITH MEMORY CONTEXT
    print("\n--- DEEP LEARNING SUMMARY (WITH BUFFER MEMORY) ---")

    context = memory.get_context()

    prompt = f"""
Previous conversation:
{context}

Now summarize this text and ensure it relates to previous summary:

{dl_text}
"""

    dl_summary = summarize(prompt)
    print(dl_summary)

    memory.add("user", dl_summary)


# TASK 6 - SUMMARY MEMORY TEST
def run_summary_memory_test():
    print("\n================ SUMMARY MEMORY =================")

    memory = []

    ml_text = """
Machine learning is a field of AI that allows computers to learn patterns from data. 
It is widely used in prediction systems, recommendation engines, and automation tasks. 
It relies on training data and algorithms to improve performance over time.
"""

    dl_text = """
Deep learning uses multi-layer neural networks to model complex patterns in data. 
It is a more advanced subset of machine learning that powers technologies like voice assistants and image recognition systems. 
It performs well with large datasets and high computational resources.
"""

    # STEP 1
    ml_summary = summarize(ml_text)
    print("\n--- ML SUMMARY ---")
    print(ml_summary)

    memory.append(ml_summary)

    # STEP 2 with summary memory
    print("\n--- DEEP LEARNING SUMMARY (WITH SUMMARY MEMORY) ---")

    summary_context = "Previous summaries:\n" + "\n".join(memory)

    prompt = f"""
{summary_context}

Now summarize this text considering previous summaries:

{dl_text}
"""

    dl_summary = summarize(prompt)
    print(dl_summary)

    memory.append(dl_summary)


# RUN BOTH TESTS
if __name__ == "__main__":
    run_buffer_memory_test()
    run_summary_memory_test()