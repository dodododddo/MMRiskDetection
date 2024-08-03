import socket

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print("Local IP address:", local_ip)
