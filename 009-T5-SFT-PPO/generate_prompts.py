from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "Here are three short sentences:\n"
    "1. It is funny.\n"
    "2. It is sad. 3. "
]

generated_sentences = []
for prompt in prompts:
  result = generator(prompt,
                     max_length=50,
                     num_return_sequences=3,
                     temperature=0.8)
  for item in result:
    print(item['generated_text'].strip())
