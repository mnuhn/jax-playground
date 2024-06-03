import prompt_db
import argparse

parser = argparse.ArgumentParser(description='add predictions to database')
parser.add_argument('--db1', dest='db1', default=None, help='db1 to read from')
parser.add_argument('--db2', dest='db2', default=None, help='db to read from')
parser.add_argument('--db_out',
                    dest='db_out',
                    default=None,
                    help='db to write predictions to')
args = parser.parse_args()

db1 = prompt_db.prompt_db(args.db1)
db2 = prompt_db.prompt_db(args.db2)
db_out = prompt_db.prompt_db(args.db_out)


def import_db(from_db, to_db):
  cursor = from_db.conn.execute("""
  SELECT
    prompts.prompt,
    c1.completion,
    c1.reward,
    c1.rule_reward,
    c1.score,
    rating1,
    c2.completion,
    c2.reward,
    c2.rule_reward,
    c2.score,
    rating2 
  FROM
    prompts
    JOIN comparisons ON prompts.id = comparisons.pid
    JOIN completions AS c1 ON comparisons.cid1 = c1.id
    JOIN completions AS c2 ON comparisons.cid2 = c2.id
  WHERE
    rating1 != rating2""")

  results = cursor.fetchall()
  for prompt, c1, c1_reward, c1_rule_reward, c1_score, r1, c2, c2_reward, c2_rule_reward, c2_score, r2 in results:
    pid = to_db.add_prompt(prompt)
    cid1 = to_db.add_completion(pid, "import", c1, c1_reward, c1_rule_reward,
                                c1_score)
    cid2 = to_db.add_completion(pid, "import", c2, c2_reward, c2_rule_reward,
                                c2_score)
    to_db.add_comparison(pid, cid1, cid2, r1, r2)

    print(pid, prompt)
    print("  ", cid1, c1, c1_reward, c1_rule_reward, c1_score)
    print("  ", cid2, c2, c2_reward, c2_rule_reward, c2_score)
    print()
  to_db.conn.commit()


import_db(db1, db_out)
import_db(db2, db_out)
