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

            similar_vectors = ""
            for i in range(num_results):
                similar_vectors += f"\n TEXT_CHUNK_{i+1}: {relevant_documents[i].page_content}"

            for i in range(num_results_2):
                similar_vectors += f"\n AYURVEDIC_TEXT_CHUNK_{i+1}: {relevant_documents_2[i].page_content}"


            completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                # Step 1: System message to restrict responses
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer only based on the provided information. "
                        "If the answer is not in the context, say 'I don't know based on the given information.'. Dont Hallucinate."
                        "Also don't write in markdown text format, write in normal structured text"
                        f"Here is relevant information retrieved from a database: {similar_vectors}"
                        f" User Query : {query}"
                        "RETURN Normal and Ayurvedic Text seperate as 2 solutions , one from normal medicine and other ayurvedic"
                    )
                }
            ]
        )
            response_text=completion.choices[0].message.content
            print(f"response_text: {response_text}")
            self.convo_history.append({"role":"user", "content":query})
            self.convo_history.append({"role":"assistant", "content":response_text})
            print("✅ Retrieval successful!")
            return response_text



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
                    {"role": "system", "content": (
                        "You are a classifier responsible for determining whether a given query requires fetching from the vector database. "
                        "Before making a decision, follow these steps:\n\n"

                        "🔹 **Step 1: Language Detection & Translation**\n"
                        "- If the query is not in English, first translate it to English while preserving its medical context.\n"
                        "- Use the translated English version for the subsequent steps.\n\n"

                        "🔹 **Step 2: Query Classification**\n"
                        "- Determine whether the query should fetch data from the vector database.\n"
                        "  - If the query is about medical conditions, symptoms, treatments, or pharmaceutical details, set `fetch_from_db = 1`.\n"
                        "  - If the query is a general conversational response (e.g., 'Thank you' or 'That was an amazing insight'), set `fetch_from_db = -1`.\n\n"

                        "🔹 **Step 3: Assigning Alpha for Hybrid Search**\n"
                        "- Compute `alpha`, a float (0 < alpha < 1), determining the weight between dense and sparse retrieval:\n"
                        "  - **General medical questions** (e.g., 'What is Skin Cancer?') → High `alpha` (~0.90) since dense retrieval is more useful.\n"
                        "  - **Treatment-based queries** (e.g., 'Give me a cure for acne and pimples') → Moderately high `alpha` (~0.75–0.80).\n"
                        "  - **Balanced queries** (e.g., 'Give me a cure for Diabetes Type 2') → Mid-range `alpha` (~0.50).\n"
                        "  - **Specific symptom/treatment queries** (e.g., 'Give me early symptoms and cure for Skin Cancer') → Lower `alpha` (~0.30–0.40), prioritizing sparse retrieval.\n"
                        "  - **Highly specific pharmaceutical queries** (e.g., 'Give me salt names for tramadol hydrochloride') → Very low `alpha` (~0.20–0.30).\n\n"

                        "🔹 **Step 4: Preparing the Refined Query for Retrieval**\n"
                        "- `refined_query` should be the translated and reformulated version of the query, ensuring it is optimized for retrieval in the vector database.\n"
                        "- If the original query was already in English, use it directly.\n"
                        "- If the query was translated, ensure `refined_query` contains the English version.\n\n"

                        "❗ **Special Cases:**\n"
                        "- For queries like 'Thank you' or 'That was an amazing insight', set `fetch_from_db = -1`, and return the original query as `refined_query`.\n"
                    )},
                    *convo_history,  # Include the last 5 messages for context
                    {"role": "user", "content": query},
                ],
                response_format=Checkpoint,  # Maintain structured response format
            )

            print(completion)
            # Parse response
            res_dict=json.loads(completion.choices[0].message.content)

            #Handle the response correctly
            if res_dict["fetch_from_db"] == -1:
                print("CONTINUING CONVO :")
                return self.normal_convo(res_dict["refined_query"])  # Use self.
            elif res_dict["fetch_from_db"]==1 and 0<= res_dict["alpha"] <= 1:
                print("FETCHING FROM DB")
                return self.fetch_from_db(res_dict["refined_query"], res_dict["alpha"])  # Use self.
            else:
                print("UNKNOWN VALUES AT CHECKPOINT, FETCHING FROM DB")
                return self.fetch_from_db(query)
```

#### **`normal_convo()`**

Handles normal conversation without database retrieval.

```python
    def normal_convo(self, query):
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    # System message to guide behavior
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Answer the user's query, but do NOT provide any medical advice."
                        )
                    },
                    # Include previous conversation history
                    *self.convo_history,
                    # Add the latest user query
                    {"role": "user", "content": query}
                ]
            )
            # Extract response text
            response_text = completion.choices[0].message.content
            # Update conversation history with user query and assistant response
            self.convo_history.append({"role": "user", "content": query})
            self.convo_history.append({"role": "assistant", "content": response_text})

            return response_text
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

