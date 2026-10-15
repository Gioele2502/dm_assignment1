import regex as re
import json
import ast
from collections import Counter

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: dict[str, int] = None):
        self.vocab = vocab
        self.merges = merges
        self.special = special_tokens if special_tokens is not None else {}
        self.token_to_id = {v: k for k, v in vocab.items()}
        
        # NUOVA RIGA: Dizionario per cercare velocemente la priorità di una coppia
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens):
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
            vocab = {int(k): ast.literal_eval(v) if isinstance(v, str) else v for k, v in vocab_raw.items()}
            
        with open(merges_path, "r", encoding="utf-8") as f:
            merges_raw = json.load(f)
            merges = [tuple(ast.literal_eval(p) for p in pair) for pair in merges_raw]
            
        return cls(vocab, merges, special_tokens)

    
    def encode(self, text):
        ids = []
        '''
        Isola i token speciali: Se hai definito dei token speciali (come <|endoftext|>), 
        il codice crea una regola per cercarli nel testo. 
        Li ordina dal più lungo al più corto ("longest match first") per evitare di spezzare per errore token simili. 
        Poi usa re.split per "affettare" il testo in una lista chiamata chunks, 
        separando in modo pulito il testo normale dai token speciali.
        '''
        if self.special:
            sorted_specials = sorted(self.special.keys(),key=len, reverse=True)
            special_pattern = "|".join(re.escape(s) for s in sorted_specials)
            chunks =re.split(f"({special_pattern})", text)
        else:
            chunks = [text]

        for chunk in chunks:
            if chunk in self.special:
                ids.append(self.special[chunk]); continue       
            
            for pre in re.findall(PAT, chunk): # pre-tokens
                # Usiamo una lista invece di una tupla per poterla modificare più facilmente
                toks = [bytes([b]) for b in pre.encode()]  
                
                while len(toks) > 1:
                    # Trova la coppia di token adiacenti con il rank (priorità) migliore
                    best_idx = None
                    best_rank = float('inf')
                    for i in range(len(toks) - 1):
                        pair = (toks[i], toks[i+1])
                        # Se la coppia esiste nelle regole e ha una priorità migliore, salvala
                        if pair in self.merge_ranks and self.merge_ranks[pair] < best_rank:
                            best_rank = self.merge_ranks[pair]
                            best_idx = i
                            
                    # Se non abbiamo trovato nessuna coppia fondibile, la parola è finita
                    if best_idx is None:
                        break
                        
                    # Fonde la coppia migliore fisicamente nella lista
                    toks = toks[:best_idx] + [toks[best_idx] + toks[best_idx+1]] + toks[best_idx+2:]
                    
                ids.extend(self.token_to_id[t] for t in toks)
        return ids

    def encode_iterable(self, iterable):
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


PAT = (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+"""
       r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def count_pretokens(corpus, special_tokens):
    pre_counts = Counter() 

    file = open(corpus, "r", encoding="utf-8")
    text = file.read()
    file.close()

    if special_tokens:
        pattern = "|".join(re.escape(s) for s in special_tokens)
        chunks = re.split(pattern, text)
    else: 
        chunks =[text]

    for chunk in chunks:
        pre_counts.update(re.findall(PAT, chunk))

    return dict(pre_counts)

def merge(token, pair):

    list_tok = []
    i = 0
    while i < len(token):
        if i < len(token)-1 and token[i] == pair[0] and token[i+1] == pair[1]:
            list_tok.append(pair[0]+pair[1])
            i += 2          # +2 perchè devo passare a due token sucessivi

        else:
            list_tok.append(token[i])
            i += 1

    return tuple(list_tok)


def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}

    # ... plus one id per special token!!!!!!!!!!!!!!!!!!!!!!!
    for token, tok_id in special_tokens.items():
        vocab[tok_id] = token.encode("utf-8")

    counts = count_pretokens(input_path, special_tokens) # {str: n}
    #trasforma 'casa' in b'casa' e poi prende ogni lettere in formato byte (es. 99 per la 'c', 97 per la 'a', ecc.)
    #output di seqs =(b'c', b'a', b's', b'a')
    seqs = {w: tuple(bytes([b]) for b in w.encode()) for w in counts}
    merges = []

    while len(vocab) < vocab_size:
        pairs = Counter()
        for w, c in counts.items():
            for a, b in zip(seqs[w], seqs[w][1:]): # prende la sequenza dallo 0 e prende la sequenza partendo dall'elemento in posizione 1
                # seqs[w] è (b'c', b'a', b's', b'a'), a e b diventeranno a ogni passo: b'c' e b'a', poi b'a' e b's', poi b's' e b'a'
                pairs[(a, b)] += c
                # Se la parola " casa" compare 100 volte, significa che anche la coppia (b'c', b'a') compare 100 volte in quel contesto.
        
        if not pairs:
            break

        best = max(pairs, key=lambda p: (pairs[p], p)) # Ordina le coppie prima in base alla loro frequenza numerica pairs[p]
        merges.append(best)
        # len(vocab) rappresenta sempre e in modo automatico il prossimo ID
        # Prende i due pezzi e li fonde fisicamente in un unico nuovo oggetto. Il risultato di b'c' + b'a' diventa semplicemente b'ca'
        vocab[len(vocab)] = best[0] + best[1]
        # Riscrive completamente il dizionario seqs unendo i due tokens più frequenti
        seqs = {w: merge(seqs[w], best) for w in seqs}
    return vocab, merges