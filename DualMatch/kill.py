import os

result = os.popen("fuser -v /dev/nvidia*").read()
results = result.split()
for pid in results:
    os.system(f"kill -9 {int(pid)}")