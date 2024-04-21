from http.server import SimpleHTTPRequestHandler
from http.server import HTTPServer
from http_handler import MyHTTPBaseHandler
from socketserver import ThreadingMixIn

import sqlite3
import threading
import argparse
import prompt_db

import viewer_common

parser = argparse.ArgumentParser(description='HTTP rating tool for SFT')
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


def retrieve_prompt():
  print("retrieve prompt")
  pid, prompt_str, reward, score = db.retrieve_prompt()

  output = f'<div style="font-size: 7vw">{prompt_str}</div>'
  output += f"<form action='/answer/{pid}/' method='get'>"
  output += '<button type="submit" style="font-size: 7vw">submit</button>'
  output += '<br/>'

  print("retrieve completions")
  completions = [(-1, "NONE", -1, -1)]
  completions.extend(db.retrieve_completions(pid))

  counter = 1
  for cid, completion, reward, score in completions:
    output += f'{counter}. reward={reward:.2f} score={score:.2f} '
    output += f'<label for="{cid}" style="font-size: 7vw">'
    output += '<div style="border-style: dashed;">'
    output += f'<input type="checkbox" id="{cid}" name="{cid}">'
    output += f'{completion}</div></label>'
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
          viewer_common.HEADER + retrieve_prompt() + viewer_common.FOOTER,
          "utf-8")
    elif self.path.startswith("/answer"):
      rest = self.path[8:]
      prompt_id = int(rest.split("/")[0])
      rest = rest.split("?")[1]
      for completion_id in rest.split("&"):
        print("add rating")
        completion_id = completion_id.replace("=on", "")
        db.add_rating(prompt_id, completion_id, rating=1.0)
      db.conn.commit()
      print("commit done")

      self.send_response(200)
      viewer_common.send_html_header(self)
      response = bytes(
          viewer_common.HEADER + "done<br/>" + retrieve_prompt() +
          viewer_common.FOOTER, "utf-8")
    if response:
      self.wfile.write(response)


if __name__ == '__main__':
  server = ThreadingSimpleServer((args.host, args.port), HttpHandler)
  print('Starting server, use <Ctrl-C> to stop')
  server.serve_forever()
