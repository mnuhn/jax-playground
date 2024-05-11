import os
import sqlite3


class prompt_db:

  def __init__(self, fn):
    self.conn = sqlite3.connect(fn, check_same_thread=False)
    self.conn.execute("""CREATE TABLE IF NOT EXISTS prompts (
                                           id integer PRIMARY KEY,
                                           prompt text NOT NULL,
                                           UNIQUE(prompt)
                                       ); """)

    self.conn.execute("""CREATE TABLE IF NOT EXISTS completions (
                                           id integer PRIMARY KEY,
                                           pid integer,
                                           name text,
                                           completion text,
                                           reward float,
                                           rule_reward float,
                                           score float,
                                           UNIQUE(pid, completion)
                                       ); """)

    self.conn.execute("""CREATE TABLE IF NOT EXISTS ratings (
                                           id integer PRIMARY KEY,
                                           pid integer,
                                           cid integer,
                                           rating float,
                                           FOREIGN KEY (pid) REFERENCES prompts(id),
                                           FOREIGN KEY (cid) REFERENCES completions(id)
                                       ); """)

    self.conn.execute("""CREATE TABLE IF NOT EXISTS comparisons (
                                           id integer PRIMARY KEY,
                                           pid integer,
                                           cid1 integer,
                                           cid2 integer,
                                           rating1 float,
                                           rating2 float,
                                           FOREIGN KEY (pid) REFERENCES prompts(id),
                                           FOREIGN KEY (cid1) REFERENCES completions(id)
                                           FOREIGN KEY (cid2) REFERENCES completions(id)
                                       ); """)

    self.conn.execute(
        """CREATE INDEX IF NOT EXISTS ratings_pid ON ratings(pid);""")
    self.conn.execute(
        """CREATE INDEX IF NOT EXISTS completions_pid ON completions(pid);""")
    self.conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_completions_completion ON completions(completion);"""
    )

    self.conn.commit()

  def get_preference_pairs_gen(self):
    cursor = self.conn.execute("""
    SELECT prompts.prompt, c1.completion, rating1, c2.completion, rating2 FROM
      prompts
      JOIN comparisons ON prompts.id = comparisons.pid
      JOIN completions AS c1 ON comparisons.cid1 = c1.id
      JOIN completions AS c2 ON comparisons.cid2 = c2.id
    WHERE
      rating1 != rating2""")

    results = cursor.fetchall()

    def gen():
      for p, c1, r1, c2, r2 in results:
        if r2 > r1:
          r1, r2 = r2, r1
          c1, c2 = c2, c1

        yield {
            "prompt_text": f'Negate:\n{p}',
            "accepted_text": f'{c1}',
            "rejected_text": f'{c2}'
        }

    return gen

  def retrieve_random_pair(self):
    print("query")
    cursor = self.conn.execute("""
		SELECT 
			p.id AS prompt_id,
			p.prompt,
			c1.id AS completion1_id,
			c1.name AS name1,
			c1.completion AS completion1_text,
			c1.score AS score1,
			c2.id AS completion2_id,
			c2.name AS name2,
			c2.completion AS completion2_text,
			c2.score AS score2,
			ABS(c1.score - c2.score) AS score_difference
		FROM 
			prompts p
		JOIN 
			completions c1 ON p.id = c1.pid
		JOIN 
			completions c2 ON p.id = c2.pid
		LEFT JOIN 
			comparisons comp ON (
				comp.pid = p.id)
		WHERE 
			c1.id <> c2.id
			AND c1.name <> c2.name
			AND c1.name = 'default'
            AND c2.name != 'not'
            AND c2.name != 'no_not'
			AND c1.completion <> c2.completion
			AND NOT EXISTS (
				SELECT 1
				FROM completions c3
				WHERE c3.completion = c2.completion AND c3.name = c1.name
			)
			AND comp.id IS NULL
		ORDER BY 
            c2.score DESC,
            c1.score DESC
		LIMIT 1""")
    pid, prompt_str, cid1, name1, completion1, score1, cid2, name2, completion2, score2, _ = cursor.fetchone(
    )
    print("done")
    print(name1, name2)

    return pid, prompt_str, cid1, completion1, score1, cid2, completion2, score2

  def retrieve_prompt(self):
    cursor = self.conn.execute("""
      SELECT prompts.id, prompt, reward, score FROM prompts, completions
      WHERE NOT EXISTS (
        SELECT 1
        FROM ratings
        WHERE prompts.id = ratings.pid
      ) AND
        prompts.id = completions.pid
      ORDER BY prompts.id DESC
      LIMIT 1;""")

    result = cursor.fetchone()
    print(result)
    return result

  def retrieve_completions(self, pid):
    cursor = self.conn.execute(
        """
      SELECT id, completion, rule_reward, score
      FROM completions
      WHERE pid = ?
      ORDER BY score DESC, rule_reward DESC, RANDOM();
      """, (pid,))

    return cursor.fetchall()

  def get_completions(self):
    cursor = self.conn.execute(
        """SELECT prompts.prompt, completions.completion, completions.score, completions.reward
       FROM prompts, completions
       WHERE completions.pid = prompts.id
       ORDER BY prompts.id""")

    return cursor.fetchall()

  def get_rated_completions(self):
    cursor = self.conn.execute("""SELECT prompts.prompt, completions.completion
       FROM prompts, completions, ratings
       WHERE
         completions.pid = prompts.id AND
         completions.id = ratings.cid AND
         ratings.rating > 0""")

    return cursor.fetchall()

  def add_comparison(self, pid, cid1, cid2, rating1, rating2):
    # always have cid1 < cid2
    if cid1 > cid2:
      cid1, cid2 = cid2, cid1
      rating1, rating2 = rating2, rating1

    cursor = self.conn.execute(
        "INSERT OR REPLACE INTO comparisons (pid, cid1, cid2, rating1, rating2) VALUES (?,?,?,?,?)",
        (
            pid,
            cid1,
            cid2,
            rating1,
            rating2,
        ))
    return cursor.lastrowid

  def add_rating(self, pid, cid, rating=1.0):
    cursor = self.conn.execute(
        "INSERT OR REPLACE INTO ratings (pid, cid, rating) VALUES (?,?,?)", (
            pid,
            cid,
            rating,
        ))
    return cursor.lastrowid

  def add_prompt(self, prompt):
    try:
      cursor = self.conn.execute("INSERT INTO prompts (prompt) VALUES (?)",
                                 (prompt,))
      return cursor.lastrowid
    except sqlite3.IntegrityError:
      cursor = self.conn.execute('SELECT id FROM prompts WHERE prompt = ?',
                                 (prompt,))
      item_id = cursor.fetchone()[0]
      return item_id  # Return the existing ID

  def add_completion(self, pid, name, completion, reward, rule_reward, score):
    try:
      cursor = self.conn.execute(
          "INSERT INTO completions (pid, name, completion, reward, rule_reward, score) VALUES (?,?,?,?,?,?)",
          (
              pid,
              name,
              completion,
              reward,
              rule_reward,
              score,
          ))
      return cursor.lastrowid
    except sqlite3.IntegrityError:
      cursor = self.conn.execute(
          'SELECT id FROM completions WHERE pid = ? AND completion = ?', (
              pid,
              completion,
          ))
      item_id = cursor.fetchone()[0]
      return item_id  # Return the existing ID
