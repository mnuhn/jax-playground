from http.server import SimpleHTTPRequestHandler
from http.server import HTTPServer
from http_handler import MyHTTPBaseHandler
from socketserver import ThreadingMixIn

import sqlite3
import threading
import argparse
import prompt_db

import viewer_common

parser = argparse.ArgumentParser(description='HTTP rating tool for PPO')
parser.add_argument('--db', dest='db', default=None, help='which db to open')
parser.add_argument('--port',
                    dest='port',
                    default=18080,
                    help='port for server')
parser.add_argument('--host',
                    dest='host',
                    default='0.0.0.0',
                    help='where to listen for requests')

args = parser.parse_args()

assert (args.db)
db = prompt_db.prompt_db(args.db)


def retrieve_random_pair():
  FONTSIZE = "3vw"
  print("retrieve prompt")
  pid, prompt_str, cid1, completion1, score1, cid2, completion2, score2 = db.retrieve_random_pair(
  )
  prompt_words = set(prompt_str.split())
  prompt_words_lower = set(prompt_str.lower().split())

  completion_words = set(completion1.lower().split() +
                         completion2.lower().split())

  prompt_str_html = ""
  space = ""
  for w in prompt_str.split():
    if w.lower() in completion_words:
      color = 'none'
    else:
      color = '#FFCCCB'

    prompt_str_html += f"{space}<span style='background-color: {color}'>{w}</span>"
    space = " "

  output = ''
  output += f"<form action='/answer/{pid}/{cid1}/{cid2}/input' method='get'>"
  output += f'<button type="submit" style="font-size: {FONTSIZE}">submit</button>'
  output += '<br/>'
  output += '<br/>'
  output += f'<div style="border-style: dashed; font-size: {FONTSIZE}; background-color: #F0F0F0">{prompt_str_html}</div>'
  output += '<br/>'

  print("retrieve completions")
  completions = [(-1, "NONE", -1, -1)]

  counter = 1
  for cid, completion, score, other in [
      (cid1, completion1, score1, completion2),
      (cid2, completion2, score2, completion1)
  ]:
    other = set(other.lower().split())
    completion_html = ""
    space = ""
    for w in completion.split():
      in_prompt = w in prompt_words
      in_prompt_lower = w in prompt_words_lower and not w in prompt_words
      in_other = w in other

      color = 'white'
      if in_prompt_lower:
        color = '#F0F0F0'
      if not in_prompt and not in_prompt_lower:
        if in_other:
          color = '#ECFFDC'
        else:
          color = 'lightyellow'

      completion_html += f"{space}<span style='background-color: {color}'>{w}</span>"
      space = " "

    completion_html = completion_html.strip()
    output += f'{counter}. score={score:.2f}'
    output += f'<label for="{cid}" style="font-size: {FONTSIZE}">'
    output += '<div style="border-style: dashed;">'
    output += f'{completion_html}'
    output += f'<input type="checkbox" id="{cid}" name="{cid}">'
    output += f'</div></label>'
    output += "<br/>"
    counter += 1

  output += "</form>"

  return output


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
  pass


class HttpHandler(MyHTTPBaseHandler):

  def do_GET(self):
    response = None
    print(self.path)
    if self.path == "/rate":
      self.send_response(200)
      viewer_common.send_html_header(self)
      response = bytes(
          viewer_common.HEADER + retrieve_random_pair() + viewer_common.FOOTER,
          "utf-8")
    elif self.path.startswith("/answer"):
      rest = self.path[8:]

      fields = rest.split("/")
      params = rest.split("?")[1]

      pid = int(fields[0])

      ratings = {}
      cid1 = int(fields[1])
      ratings[cid1] = 0.0
      cid2 = int(fields[2])
      ratings[cid2] = 0.0

      for cid in params.split("&"):
        cid = cid.replace("=on", "")
        if cid == "":
          continue
        cid = int(cid)
        ratings[cid] = 1.0

      print(f"add rating for {cid1} vs {cid2}:", ratings)
      db.add_comparison(pid, cid1, cid2, ratings[cid1], ratings[cid2])
      db.conn.commit()
      print("commit done")

      self.send_response(200)
      viewer_common.send_html_header(self)
      response = bytes(
          viewer_common.HEADER + "done<br/>" + retrieve_random_pair() +
          viewer_common.FOOTER, "utf-8")
    if response:
      self.wfile.write(response)


if __name__ == '__main__':
  server = ThreadingSimpleServer((args.host, args.port), HttpHandler)
  print('Starting server, use <Ctrl-C> to stop')
  server.serve_forever()
