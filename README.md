# DermaChatBot Project Documentation

## **Table of Contents**

1. [Problem Statement](#problem-statement)
2. [Project Overview](#project-overview)
3. [Architecture and Workflow](#architecture-and-workflow)
4. [Code Breakdown](#code-breakdown)
   - [DermaChatBot Class](#dermachatbot-class)
   - [Functions](#functions)
5. [Deployment with Flask](#deployment-with-flask)
6. [Sample Results](#sample-results)
7. [How to Run](#how-to-run)

---

## **Problem Statement**

The goal of this project is to build an AI-powered Dermatology assistant chatbot that:

- Provides medical and Ayurvedic solutions for skin-related queries.
- Utilizes hybrid search with dense and sparse retrieval models.
- Handles conversational history efficiently while interacting with users.
- Integrates Pinecone for vector-based document search and OpenAI GPT models for response generation.

---

## **Project Overview**

DermaChatBot is designed to assist users by retrieving both modern medical and Ayurvedic solutions for their skincare-related questions. The system fetches relevant information from a vector database and generates responses using GPT models.

Key features include:

- Hybrid search using Pinecone.
- Dense and sparse encoders for enhanced search accuracy.
- Checkpoint-based query classification.
- Integration with OpenAI GPT models for contextual answers.
- Flask-based API for user interaction.

---

## **Architecture and Workflow**

1. **User Interaction:** Users interact with the chatbot by sending queries.
2. **Query Processing:**
   - Checkpoint system classifies whether to fetch from the vector database or continue normal conversation.
3. **Data Retrieval:** Relevant documents are retrieved from Pinecone using a hybrid search strategy.
4. **Response Generation:** The chatbot generates responses using OpenAI GPT models based on the retrieved data.
5. **Conversation Management:** Conversation history is maintained for context-aware responses.

---

## **Code Breakdown**

### **DermaChatBot Class**

The main class for managing the chatbot operations.

#### **Initialization (`__init__`)**

```python
class DermaChatBot:
    def __init__(self, openai_key, pinecone_api_key, index_name, namespace, namespace_2):
        self.convo_history = []
        self.openai_client = OpenAI(api_key=openai_key)
        self.index_name = index_name
        self.namespace = namespace
        self.namespace_2 = namespace_2
        self.alpha = 0.5
        self.MAX_HISTORY = 10
        try:
            models = self.openai_client.models.list()
            print("✅ OpenAI client initialized successfully!")
        except Exception as e:
            print("❌ OpenAI client initialization failed!", str(e))

        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(self.index_name)
            print("✅ Pinecone client initialized successfully!")
        except Exception as e:
            print("❌ Pinecone client initialization failed!", str(e))
```

### **Functions**

#### **`start_convo()`**

Starts the conversation by returning a greeting.

```python
    def start_convo(self):
        start_query = "Hi, I am your Dermatologist assistant. How can I help you today?"
        self.convo_history.append({"role": "assistant", "content": start_query})
        return start_query
```

#### **`fetch_from_db()`**

Fetches relevant documents from Pinecone.

```python
    def fetch_from_db(self, query: str, alpha=0.5, top_k=5):
        try:
            retriever = PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=bm25_encoder_new,
                alpha=alpha,
                top_k=top_k,
                index=self.index,
                namespace=self.namespace
            )
            retriever_2 = PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=bm25_encoder_ayurved,
                alpha=alpha,
                top_k=top_k,
                index=self.index,
                namespace=self.namespace_2
            )
            relevant_documents = retriever.invoke(query)
            relevant_documents_2 = retriever_2.invoke(query)
```

#### **`checkpoint_check()`**

Determines whether a query requires database retrieval or normal conversation.

```python
    def checkpoint_check(self, query):
        class Checkpoint(BaseModel):
            fetch_from_db: int
            alpha: float
            refined_query: str

        completion = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a classifier responsible for query classification."},
                {"role": "user", "content": query}
            ],
            response_format=Checkpoint
        )
        res_dict = json.loads(completion.choices[0].message.content)
        if res_dict["fetch_from_db"] == -1:
            return self.normal_convo(res_dict["refined_query"])
        elif res_dict["fetch_from_db"] == 1:
            return self.fetch_from_db(res_dict["refined_query"], res_dict["alpha"])
```

#### **`normal_convo()`**

Handles normal conversation without database retrieval.

```python
    def normal_convo(self, query):
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the user's query but do NOT provide any medical advice."},
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content
```

---

## **Deployment with Flask**

The project includes a Flask application for easy interaction.

### **Endpoints**

- **`/initialize`**: Initializes the chatbot with API keys and configuration.
- **`/start_convo`**: Starts the conversation.
- **`/continue_convo`**: Continues the conversation with user input.

Example initialization route:

```python
@app.route("/initialize", methods=["POST"])
def initialize_bot():
    global DermaBot
    data = request.get_json()
    openai_key = data.get("openai_key")
    pinecone_key = data.get("pinecone_key")
    DermaBot = DermaChatBot(openai_key, pinecone_key, "medicalchatbothybrid", "hybrid_namespace", "hybrid_namespace_ayurved")
    return jsonify({"message": "✅ DermaChatBot initialized successfully!"})
```

---

## **Sample Results**

- **User Query:** "What are the best treatments for acne?"
- **Normal Medicine:** "Topical treatments such as benzoyl peroxide..."
- **Ayurvedic Solution:** "Apply turmeric paste and use neem extracts..."

---

## **How to Run**

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install openai pinecone-client flask
   ```
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Initialize the chatbot by sending a POST request to `/initialize` with API keys.
5. Start the conversation by accessing `/start_convo`.

---

## **Conclusion**

The DermaChatBot project showcases the integration of advanced NLP models and vector search techniques to deliver dual medical and Ayurvedic solutions. The system demonstrates robust query handling, conversational context retention, and efficient database retrieval, making it a powerful tool for dermatological assistance.

