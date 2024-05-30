import prompt_db

prompt = None
examples = []


db = prompt_db.prompt_db("./ollamma.db")

for l in open("./ollamma.txt"):
  l = l.strip()
  print(l)
  if l == "":
    if prompt != None:
      pid = db.add_prompt(prompt)
      for e in examples:
        db.add_completion(pid, "llama", e, 0.0, 0.0, 0.0)

    prompt = None
    continue

  if prompt == None:
    prompt = l
    examples = []
    continue

  examples += [l]
db.conn.commit()
