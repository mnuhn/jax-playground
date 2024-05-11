from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "Here are four things that happened today:\n"
    "1. My uncle went for a long walk outside.\n"
    "2. It was raining a lot outside.\n"
    "3. The stocks went up in value a lot.\n"
    "4. "
]

generated_sentences = []
for prompt in prompts:
  result = generator(prompt,
                     max_length=20,
                     num_return_sequences=3,
                     temperature=0.8)
  for item in result:
    print(item['generated_text'].strip())
