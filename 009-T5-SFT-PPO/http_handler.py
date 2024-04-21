from http.server import BaseHTTPRequestHandler

import shutil
import datetime
import os
import email
import mimetypes
import urllib
import posixpath
from http import HTTPStatus


class MyHTTPBaseHandler(BaseHTTPRequestHandler):

  def __init__(self, *args, directory=None, **kwargs):
    if directory is None:
      directory = os.getcwd()
    self.directory = directory
    super().__init__(*args, **kwargs)

  def copyfile(self, source, outputfile):
    """Copy all data between two file objects.
    The SOURCE argument is a file object open for reading
    (or anything with a read() method) and the DESTINATION
    argument is a file object open for writing (or
    anything with a write() method).
    The only reason for overriding this would be to change
    the block size or perhaps to replace newlines by CRLF
    -- note however that this the default server uses this
    to copy binary data as well.
    """
    shutil.copyfileobj(source, outputfile)

  def translate_path(self, path):
    """Translate a /-separated PATH to the local filename syntax.
    Components that mean special things to the local file system
    (e.g. drive or directory names) are ignored.  (XXX They should
    probably be diagnosed.)
    """
    # abandon query parameters
    path = path.split('?', 1)[0]
    path = path.split('#', 1)[0]
    # Don't forget explicit trailing slash when normalizing. Issue17324
    trailing_slash = path.rstrip().endswith('/')
    try:
      path = urllib.parse.unquote(path, errors='surrogatepass')
    except UnicodeDecodeError:
      path = urllib.parse.unquote(path)
    path = posixpath.normpath(path)
    words = path.split('/')
    words = filter(None, words)
    path = self.directory
    for word in words:
      if os.path.dirname(word) or word in (os.curdir, os.pardir):
        # Ignore components that are not a simple file/directory name
        continue
      path = os.path.join(path, word)
    if trailing_slash:
      path += '/'
    return path

  def guess_type(self, path):
    """Guess the type of a file.
    Argument is a PATH (a filename).
    Return value is a string of the form type/subtype,
    usable for a MIME Content-type header.
    The default implementation looks the file's extension
    up in the table self.extensions_map, using application/octet-stream
    as a default; however it would be permissible (if
    slow) to look inside the data to make a better guess.
    """

    base, ext = posixpath.splitext(path)
    if ext in self.extensions_map:
      return self.extensions_map[ext]
    ext = ext.lower()
    if ext in self.extensions_map:
      return self.extensions_map[ext]
    else:
      return self.extensions_map['']

  if not mimetypes.inited:
    mimetypes.init()  # try to read system mime.types
  extensions_map = mimetypes.types_map.copy()
  extensions_map.update({
      '': 'application/octet-stream',  # Default
      '.py': 'text/plain',
      '.c': 'text/plain',
      '.h': 'text/plain',
  })

  def send_head(self):
    """Common code for GET and HEAD commands.
    This sends the response code and MIME headers.
    Return value is either a file object (which has to be copied
    to the outputfile by the caller unless the command was HEAD,
    and must be closed by the caller under all circumstances), or
    None, in which case the caller has nothing further to do.
    """
    path = self.translate_path(self.path)
    f = None
    if os.path.isdir(path):
      parts = urllib.parse.urlsplit(self.path)
      if not parts.path.endswith('/'):
        # redirect browser - doing basically what apache does
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        new_parts = (parts[0], parts[1], parts[2] + '/', parts[3], parts[4])
        new_url = urllib.parse.urlunsplit(new_parts)
        self.send_header("Location", new_url)
        self.end_headers()
        return None
      for index in "index.html", "index.htm":
        index = os.path.join(path, index)
        if os.path.exists(index):
          path = index
          break
      else:
        return self.list_directory(path)
    ctype = self.guess_type(path)
    try:
      f = open(path, 'rb')
    except OSError:
      self.send_error(HTTPStatus.NOT_FOUND, "File not found")
      return None

    try:
      fs = os.fstat(f.fileno())
      # Use browser cache if possible
      if ("If-Modified-Since" in self.headers and
          "If-None-Match" not in self.headers):
        # compare If-Modified-Since and time of last file modification
        try:
          ims = email.utils.parsedate_to_datetime(
              self.headers["If-Modified-Since"])
        except (TypeError, IndexError, OverflowError, ValueError):
          # ignore ill-formed values
          pass
        else:
          if ims.tzinfo is None:
            # obsolete format with no timezone, cf.
            # https://tools.ietf.org/html/rfc7231#section-7.1.1.1
            ims = ims.replace(tzinfo=datetime.timezone.utc)
          if ims.tzinfo is datetime.timezone.utc:
            # compare to UTC datetime of last modification
            last_modif = datetime.datetime.fromtimestamp(
                fs.st_mtime, datetime.timezone.utc)
            # remove microseconds, like in If-Modified-Since
            last_modif = last_modif.replace(microsecond=0)

            if last_modif <= ims:
              self.send_response(HTTPStatus.NOT_MODIFIED)
              self.end_headers()
              f.close()
              return None

      self.send_response(HTTPStatus.OK)
      self.send_header("Content-type", ctype)
      self.send_header("Content-Length", str(fs[6]))
      self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
      self.end_headers()
      return f
    except:
      f.close()
      raise

  def do_HEAD(self):
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
    self.send_header('Pragma', 'no-cache')
    self.send_header('Expires', '0')
    self.send_header('Content-type', 'text/html')
    self.end_headers()

  def send_status(self, what):
    self.send_response(200)
    self.send_header('Content-type', 'text/html')
    self.end_headers()
    return bytes(str(what), 'utf-8')
