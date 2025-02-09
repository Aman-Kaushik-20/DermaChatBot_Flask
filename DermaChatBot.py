from openai import OpenAI
from langchain_community.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever
from pinecone import Pinecone
from EmbeddingModels import embeddings, bm25_encoder_new, bm25_encoder_ayurved
from pydantic import BaseModel
import json


class DermaChatBot:
    def __init__(self, openai_key, pinecone_api_key, index_name, namespace, namespace_2): ## Ensure parameters are strings
        self.convo_history=[]
        self.openai_client=OpenAI(api_key=openai_key)
        self.index_name = index_name   # "medicalchatbothybrid"
        self.namespace=namespace # hybrid_namespace_ayurved
        self.namespace_2=namespace_2       # "hybrid_namespace"
        self.alpha=0.5
        self.MAX_HISTORY=10
        try:
            # Make a simple API call to verify initialization
            models = self.openai_client.models.list()
            print("✅ OpenAI client initialized successfully!")
            print("Available models:", [model.id for model in models.data])
        except Exception as e:
            print("❌ OpenAI client initialization failed!")
            print("Error:", str(e))

        try:
            # Initialize Pinecone client
            self.pc= Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(self.index_name)            
            # Test the connection by listing indexes
            indexes = self.pc.list_indexes()
            
            print("✅ Pinecone client initialized successfully!")
            print("Available indexes:", indexes)

        except Exception as e:
            print("❌ Pinecone client initialization failed!")
            print("Error:", str(e))

    def start_convo(self):
            start_query="Hi, I am your Dermatologist assistant. How can I help you today?"
            self.convo_history.append({"role":"assistant", "content":start_query})
            return start_query
    
    def fetch_from_db(self, query:str, alpha=0.5, top_k=5):
        print("query :", query)
        print(f" type :{type(query)}")
        try:
            # Validate input types
            print("Check str")
            if not isinstance(query, str):
                print("Give query as str ")
                raise ValueError(" Error: 'query' must be a string!")
            if not isinstance(alpha, (int, float)):  # Allow both int and float for alpha
                raise ValueError(" Error: 'alpha' must be an integer or float!")
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(" Error: 'top_k' must be a positive integer!")
            
            print("Dtype Check Complete")

            # Ensure conversation history doesn't exceed 10 messages (adjustable)
            if len(self.convo_history) > self.MAX_HISTORY:
                self.convo_history = self.convo_history[-self.MAX_HISTORY:]

            # Initialize the retriever
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

            print("START FETCHING FROM DB !")

            # Fetch relevant documents
            relevant_documents = retriever.invoke(query)
            relevant_documents_2 = retriever_2.invoke(query)

            
            print("FETCH FROM DB COMPLETE !")

            # Handle case where fewer documents are returned
            num_results = min(top_k, len(relevant_documents))
            num_results_2 = min(top_k, len(relevant_documents))
       

            # Construct the response string
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

        except ValueError as ve:
            print(f"Value Error: {ve}")
        except Exception as e:
            print(f" Unexpected Error: {str(e)}")
        
        return " An error occurred while fetching data from the database."
           
                   
              
    def normal_convo(self, query):
        try:
            # Validate input type
            if not isinstance(query, str):
                raise ValueError("❌ Error: 'query' must be a string!")

            # Ensure conversation history doesn't exceed 10 messages (adjustable)
            if len(self.convo_history) > self.MAX_HISTORY:
                self.convo_history = self.convo_history[-self.MAX_HISTORY:]

            # Generate response from OpenAI
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

        except ValueError as ve:
            return f"❌ Value Error: {ve}"
        except Exception as e:
            return f"❌ Unexpected Error: {str(e)}"


    def checkpoint_check(self, query):
        try:
            # Define response schema
            class Checkpoint(BaseModel):
                fetch_from_db: int
                alpha: float
                refined_query: str

            # Ensure query is a string
            if not isinstance(query, str):
                raise ValueError("❌ Error: 'query' must be a string!")

            # Ensure conversation history doesn't exceed 5 messages
            convo_history = self.convo_history[-5:] if len(self.convo_history) >= 5 else self.convo_history

            # Make OpenAI API call
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
        

        except ValueError as ve:
            return f"❌ Value Error: {ve}"
        except Exception as e:
            return f"❌ Unexpected Error: {str(e)}"