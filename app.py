from flask import Flask, request, jsonify
from DermaChatBot import DermaChatBot


app=Flask(__name__)

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Derma Chatbot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f3f4f6;
                text-align: center;
                padding: 50px;
            }
            .container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
            }
            h1 {
                color: #4CAF50;
            }
            p {
                color: #333;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌿 Welcome to DERMA CHATBOT! 🌿</h1>
            <p>Your AI-powered dermatologist assistant is here to help with all your skincare and medical queries. Ask away!</p>
        </div>
    </body>
    </html>
    """
# Global instance to store DermaBot
DermaBot = None  

@app.route("/initialize", methods=["POST"])
def initialize_bot():
    global DermaBot
    
    data = request.get_json()
    openai_key = data.get("openai_key")
    pinecone_key = data.get("pinecone_key")
    index_name = data.get("index_name", "medicalchatbothybrid")
    namespace = data.get("namespace", "hybrid_namespace")
    namespace_2 = data.get("namespace_2", "hybrid_namespace_ayurved")

    # Validate required parameters
    if not openai_key or not pinecone_key:
        return jsonify({"error": "Missing required API keys!"}), 400

    # Initialize DermaChatBot
    DermaBot = DermaChatBot(openai_key, pinecone_key, index_name, namespace, namespace_2)

    return jsonify({"message": "✅ DermaChatBot initialized successfully!"})

@app.route("/start_convo", methods=["GET"])
def start_conversation():
    global DermaBot
    
    if DermaBot is None:
        return jsonify({"error": "❌ DermaChatBot is not initialized. Call /initialize first!"}), 400
    
    # Start the conversation and return the first response
    return jsonify({"response": DermaBot.start_convo()})


@app.route("/continue_convo", methods=["POST"])
def continue_conversation():
    """Continue the chatbot conversation by passing a user query."""
    global DermaBot
    
    if DermaBot is None:
        return jsonify({"error": "❌ DermaChatBot is not initialized. Call /initialize first!"}), 400
    
    data = request.get_json()
    input_query = data.get("query")

    if not input_query:
        return jsonify({"error": "Missing input query!"}), 400

    # Call the chatbot function to process the query
    response = DermaBot.checkpoint_check(input_query)

    return jsonify({"response": response})


@app.route("/get_history", methods=["GET"])
def get_history():
    """Return the conversation history of the chatbot."""
    global DermaBot
    
    if DermaBot is None:
        return jsonify({"error": "❌ DermaChatBot is not initialized. Call /initialize first!"}), 400

    return jsonify({"conversation_history": DermaBot.convo_history})


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)