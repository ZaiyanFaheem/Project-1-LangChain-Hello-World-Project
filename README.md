# Project-1-LangChain-Hello-World-Project
# Install required libraries
!pip install -U -q langchain langchain-google-genai

# Import necessary modules
import langchain_google_genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import json

# Replace with your actual API key
api_key = "AIzaSyBy7C2FWsiLqWrpCAwqQODgol9ZhnzwK1c"

# Initialize Google Generative AI model
llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.0-flash-exp",
    temperature=0.7,  # Adjust randomness of responses
    max_output_tokens=100  # Limit response length
)

# === 1. Experiment with Prompts ===
response = llm.invoke("Who is the founder of Pakistan?")
print("Single Prompt Response:", response)

response2 = llm.invoke("Write a short poem about technology.")
print("\nPoem Prompt Response:", response2)

# === 2. Add Memory for Multi-turn Interaction ===
# Initialize conversation memory
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
)

# Multi-turn interaction example
print("\nConversation 1:", conversation.run("Who is the founder of Pakistan?"))
print("Conversation 2:", conversation.run("When was he born?"))
print("Conversation 3:", conversation.run("Tell me more about his achievements."))

# === 3. Integrate Tools ===
# Define a custom tool for getting weather
def get_weather(location):
    # Replace with real API integration if needed
    return f"The weather in {location} is sunny and warm."

weather_tool = Tool(
    name="WeatherTool",
    func=get_weather,
    description="Provides the current weather for a given location.",
)

# Using the weather tool
weather_response = weather_tool.run("Karachi")
print("\nWeather Tool Response:", weather_response)

# === 4. Explore Gemini Model Features ===
# Custom invocation with model parameters adjusted
custom_response = llm.invoke("Explain quantum computing in simple terms.")
print("\nCustom Prompt Response:", custom_response)

# === 5. Use LangChain's Chain Feature ===
# Define a custom prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in detail.",
)

# Create an LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
chained_response = chain.run("machine learning")
print("\nChained Response:", chained_response)

# === 6. Save and Load Responses ===
# Save response to a JSON file
# Save response to a JSON file
# Save response to a JSON file
response_data = {
    "single_prompt_response": str(response),  # Convert to string
    "poem_prompt_response": str(response2),   # Convert to string
    "conversation": str(memory.load_memory_variables({})),  # Convert to string
    "weather_response": str(weather_response),  # Convert to string
    "custom_response": str(custom_response),    # Convert to string
    "chained_response": str(chained_response),  # Convert to string
}

# Save the responses into a JSON file
with open("response_data.json", "w") as f:
    json.dump(response_data, f)

print("\nResponses saved to response_data.json")


# Load responses from the JSON file
with open("response_data.json", "r") as f:
    loaded_responses = json.load(f)
print("\nLoaded Responses:", loaded_responses)
