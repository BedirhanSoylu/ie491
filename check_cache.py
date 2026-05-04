import json
from pathlib import Path

# Read detection output
try:
    detect = json.loads(Path(".graphify_detect_output.json").read_text())
except:
    print("Cannot read detection output")
    exit(1)

all_files = []
for file_list in detect.get("files", {}).values():
    if file_list:
        all_files.extend(file_list)

print(f"Total files to extract: {len(all_files)}")
print(f"Files: {all_files}")

# For now, all files are uncached (graphify cache might not be available on first run)
Path(".graphify_uncached.txt").write_text("\n".join(all_files))
Path(".graphify_cached.json").write_text(json.dumps({"nodes": [], "edges": [], "hyperedges": []}))
print(f"Uncached: {len(all_files)} files need extraction")
