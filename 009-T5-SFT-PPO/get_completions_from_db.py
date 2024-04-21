import prompt_db
import argparse
#import rulebased_reward_model as reward_model

parser = argparse.ArgumentParser(description='write model predictions to db')
parser.add_argument('--db', dest='db', default=None, help='which db to open')
parser.add_argument('--rated_only',
                    dest='rated_only',
                    default=False,
                    help='which db to open')
args = parser.parse_args()

db = prompt_db.prompt_db(args.db)

if args.rated_only:
  for prompt, completion in db.get_rated_completions():
    output_str = prompt.replace(" ", "_") + " " + completion.replace(" ", "_")
    print(output_str)
else:
  old_prompt = None
  for prompt, completion, score, reward in db.get_completions():
    output_str = prompt.replace(" ", "_") + " " + completion.replace(
        " ", "_") + " " + str(score) + " " + str(reward)
    if old_prompt != prompt:
      print(output_str)
    old_prompt = prompt
