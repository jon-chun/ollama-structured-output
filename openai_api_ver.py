from openai import OpenAI
import openai
from pydantic import BaseModel

# Initialize OpenAI client with local Ollama endpoint
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


class NER(BaseModel):
    organizations: list[str]
    products: list[str]
    people: list[str]
    locations: list[str]

CONTEXT = """
According to the third-party benchmarking organizer David Jones at Artificial Analysis, Claude 3.5 Haiku "has a lower latency compared to average, taking 0.80s to receive the first token (TTFT)," yet "is slower compared to average, with a output speed of 65.1 tokens per second."

The release — which hasn't been officially announced — comes on the heels of major updates from Anthropic's AI rivals OpenAI and Google, which have also shipped new models to general availability in their chatbots as the year winds down, namely OpenAI's o1 and o1-mini models and Google's Gemini 2.

The question for Anthropic is whether customers will be impressed enough with Claude 3.5 Haiku's performance to sign up for its Pro tier — or to continue using it instead of some of these other advanced and fast rivals.

Claude 3.5 Haiku is accessible through the Claude Chatbot
As the fastest and most cost-effective model in Anthropic's lineup, Claude 3.5 Haiku excels in real-time tasks such as processing large datasets, analyzing financial documents, and generating outputs from long-context information.

It features a 200,000-token context window — more than the 128,000-token window on OpenAI's GPT-4 and GPT-4o — allowing it to handle extensive input with ease.

On the Claude chatbot, Haiku brings functionality that enhances its versatility. Users can analyze images and file attachments, making it useful for multimedia tasks and workflows involving large document sets.

Haiku also integrates with Claude Artifacts, the interactive sidebar first introduced in June 2024. Artifacts provides a dedicated workspace for manipulating and refining AI-generated content in real time, including running full apps. In my test of Artifacts with Haiku this morning, it was able to code a fully playable version of Pong in less than a minute:
"""
SYSTEM_PROMPT = """You are a precise named entity recognition system. Your task is to carefully extract:
- Organizations: Company names, institutions, and organizations
- Products: Software, models, tools, and other products
- People: Names of individuals
- Locations: Geographic locations and places
Be thorough but avoid duplicates and only include clearly mentioned entities. If you're unsure about an entity, err on the side of exclusion."""

try:
    response = client.beta.chat.completions.parse(
        temperature=0,
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CONTEXT}
        ],
        response_format=NER,
    )

    # Check the response
    message = response.choices[0].message
    if message.parsed:
        ner = message.parsed
        
        # Print results
        print("\nOrganizations:")
        for org in ner.organizations:
            print(f"  - {org}")

        print("\nProducts:")
        for product in ner.products:
            print(f"  - {product}")

        print("\nPeople:")
        for person in ner.people:
            print(f"  - {person}")

        print("\nLocations:")
        for location in ner.locations:
            print(f"  - {location}")
    elif message.refusal:
        print("Model refused to provide structured output:", message.refusal)

except Exception as e:
    if isinstance(e, openai.LengthFinishReasonError):
        print("Too many tokens: ", e)
    else:
        print("Error:", e)
