from google import genai

client = genai.Client()
 
# Build with 3.8 Flash using tunable thinking effort
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=""Review this deployment script and identify any security vulnerabilities.",
)

print(interaction.output_text)
