import os
import json
import time
import psutil
from lm.tokenizer import train_bpe

def main():
    #input_file = "data/tinystories_train.txt"
    input_file = "data/tinystories_train_200mb.txt"
    vocab_size = 10000
    special_tokens = {"<|endoftext|>": 256}
    
    os.makedirs("tok", exist_ok=True)
    
    print("Inizio addestramento BPE")
    start_time = time.time()
    
    #avvio
    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)
    
    end_time = time.time()
    
    # RAM
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"Addestramento completato in {end_time - start_time:.2f} secondi.")
    print(f"Picco RAM stimato: {peak_ram_mb:.2f} MB")
    
    #vocab.json (converte bytes in stringhe per poter essere lette da ast.literal_eval)
    vocab_to_save = {str(k): str(v) for k, v in vocab.items()}
    with open("tok/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, indent=2)
        
    #merges.txt
    merges_to_save = [[str(a), str(b)] for a, b in merges]
    with open("tok/merges.txt", "w", encoding="utf-8") as f:
        json.dump(merges_to_save, f, indent=2)
        

if __name__ == "__main__":
    main()