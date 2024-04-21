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
                                           completion text,
                                           reward float,
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
        if r1 < r2:
          r1, r2 = r2, r1
          c1, c2 = c2, c1

        yield {
            "source_text": f'Negate:\n{p}',
            "rejected_text": c2,
            "accepted_text": c1
        }

    return gen

  def retrieve_random_pair(self):
    cursor = self.conn.execute("""
      SELECT prompts.id, prompt FROM prompts
      WHERE (
        SELECT COUNT(*)
        FROM completions
        WHERE prompts.id = completions.pid
        GROUP BY completions.pid
      ) >= 2
      ORDER BY RANDOM() LIMIT 1;""")
    pid, prompt_str = cursor.fetchone()

    cursor = self.conn.execute(
        """
      SELECT id, completion, score FROM completions
      WHERE completions.pid = ?
      ORDER BY score + RANDOM()
      LIMIT 2;""", (pid,))

    cid1, completion1, score1 = cursor.fetchone()
    cid2, completion2, score2 = cursor.fetchone()

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
      ORDER BY prompts.id -- completions.score DESC
      LIMIT 1;""")

    result = cursor.fetchone()
    print(result)
    return result

  def retrieve_completions(self, pid):
    cursor = self.conn.execute(
        """
      SELECT id, completion, reward, score FROM completions WHERE pid = ? ORDER BY completions.score DESC, reward  DESC, RANDOM();
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

  def add_completion(self, pid, completion, reward, score):
    try:
      cursor = self.conn.execute(
          "INSERT INTO completions (pid, completion, reward, score) VALUES (?,?,?,?)",
          (
              pid,
              completion,
              reward,
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
