import time
import os
from lm.tokenizer import Tokenizer

special_tokens = {"<|endoftext|>": 256}
tok = Tokenizer.from_files("tok/vocab.json", "tok/merges.txt", special_tokens)

# 1. Token più lungo e prime 20 merges
longest_token = max(tok.vocab.values(), key=len)
print(f"--- METRICHE VOCABOLARIO ---")
print(f"Token più lungo: {longest_token} (Lunghezza: {len(longest_token)})")
print("Prime 20 fusioni (merges):")
for i, m in enumerate(tok.merges[:20]):
    print(f"{i+1}: {m}")

print(f"\n--- INIZIO CODIFICA (Attendi...) ---")

# 2. Operazione UNICA: Leggiamo e codifichiamo il file una sola volta misurando il tempo
file_path = "data/tinystories_valid.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

start_time = time.time()
tokens = tok.encode(text) 
end_time = time.time() - start_time

# 3. Calcolo Byte-Per-Token usando i token appena generati
num_bytes = len(text.encode('utf-8'))
num_tokens = len(tokens)
bpt = num_bytes / num_tokens if num_tokens > 0 else 0

print(f"\n--- RISULTATI ---")
print(f"BPT su TinyStories Valid: {bpt:.2f}")

# 4. Calcolo Velocità
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Velocità di codifica: {file_size_mb / end_time:.4f} MB/s")
print(f"Tempo totale (Valid): {end_time:.2f} s")