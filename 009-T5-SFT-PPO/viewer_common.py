HEADER = """<head>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
         """

FOOTER = "</p>"


def send_html_header(handler):
  handler.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
  handler.send_header('Pragma', 'no-cache')
  handler.send_header('Expires', '0')
  handler.send_header('Content-type', 'text/html')
  handler.end_headers()

  content = '''<html>
               <head>
               <title>Title goes here.</title>
               </head>'''
  handler.wfile.write(bytes(content, 'UTF-8'))
