import http.server, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
http.server.SimpleHTTPRequestHandler.extensions_map[".wasm"] = "application/wasm"
http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=8000)
