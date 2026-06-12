
import os

def load_c_files(repo_path):
    code_chunks = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".c"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf8", errors="ignore") as f:
                    code_chunks.append({
                        "file": file,
                        "code": f.read()
                    })
    return code_chunks
