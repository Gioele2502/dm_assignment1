import math
import torch
import torch.nn as nn
import numpy as np

# sezione 4.1

class Linear(nn.Module):
    """Layer Lineare senza bias.
    Pesi W in R^(d_out x d_in). Moltiplicazione tramite einsum.
    Inizializzazione N(0, sigma^2) con sigma^2 = 2 / (d_in + d_out) troncata a +-3*sigma.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Salvavita per modelli enormi (es. Llama-3 cfg1): se supera la soglia, usa 'meta' per non esaurire la RAM
        num_params = in_features * out_features
        device = 'meta' if num_params > 15_000_000 else None
        
        # Matrice dei pesi con convenzione W in R^(d_out x d_in)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        
        # Inizializzazione Truncated Normal (eseguita solo se non siamo su 'meta')
        if self.weight.device.type != 'meta':
            std = math.sqrt(2.0 / (in_features + out_features))
            nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i, oi -> ...o", x, self.weight)


class Embedding(nn.Module):
    """Lookup table per token interi.
    Inizializzazione N(0, 1) troncata a +-3.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        num_params = num_embeddings * embedding_dim
        device = 'meta' if num_params > 15_000_000 else None
        
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device))

        if self.weight.device.type != 'meta':
            nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]


class RMSNorm(nn.Module):
    """Root Mean Square Normalization.
    Statistiche calcolate in fp32 per stabilità numerica.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        y = (x_fp32 / rms) * self.g.float()
        return y.to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU: W2(SiLU(W1 x) * W3 x) con SiLU(z) = z * sigma(z).
    Se d_ff non è fornito, viene calcolato come (8/3)*d_model arrotondato al multiplo di 64 più vicino.
    """
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        if d_ff is None:
            raw_d_ff = int(8 / 3 * d_model)
            d_ff = 64 * ((raw_d_ff + 32) // 64)
            
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        silu_gate = gate * torch.sigmoid(gate)
        return self.w2(silu_gate * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        i = torch.arange(0, d_k, 2).float()
        inv_freq = 1.0 / (theta ** (i / d_k))
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        
        return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax numericamente stabile (sottrae il massimo)."""
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Scaled Dot-Product Attention con maschera booleana (True = attend).
    Softmax calcolata in fp32.
    """
    d_k = q.shape[-1]
    scores = torch.einsum("...qd, ...kd -> ...qk", q, k) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        
    attn_weights = softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.einsum("...qk, ...kd -> ...qd", attn_weights, v)


class MultiheadSelfAttention(nn.Module):
    """Grouped-Query Attention (GQA) causale di default."""
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int | None = None, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.rope = rope
        
        self.d_k = d_model // num_heads
        
        self.q_proj = Linear(d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(d_model, self.num_kv_heads * self.d_k)
        self.v_proj = Linear(d_model, self.num_kv_heads * self.d_k)
        self.out_proj = Linear(self.num_heads * self.d_k, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        *lead, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(*lead, seq_len, self.num_heads, self.d_k)
        k = self.k_proj(x).view(*lead, seq_len, self.num_kv_heads, self.d_k)
        v = self.v_proj(x).view(*lead, seq_len, self.num_kv_heads, self.d_k)
        
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            pos = token_positions.unsqueeze(-1)
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            
        if self.num_kv_heads < self.num_heads:
            num_repeat = self.num_heads // self.num_kv_heads
            k = torch.repeat_interleave(k, num_repeat, dim=-2)
            v = torch.repeat_interleave(v, num_repeat, dim=-2)
            
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        
        if mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(-3, -2).contiguous().view(*lead, seq_len, -1)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Blocco Transformer con architettura Pre-Norm."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_kv_heads: int | None = None, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.rms_1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, num_kv_heads, rope)
        self.rms_2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.attn(self.rms_1(x), token_positions=token_positions, mask=mask)
        y = h + self.mlp(self.rms_2(h))
        return y


class TransformerLM(nn.Module):
    """Modello Linguistico Transformer completo (2024 architecture)."""
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, num_kv_heads: int | None = None, rope_theta: float = 10000.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        
        self.tok_embeddings = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, num_kv_heads, self.rope)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.output_head = Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        *lead, seq_len = ids.shape
        token_positions = torch.arange(seq_len, device=ids.device)
        
        x = self.tok_embeddings(ids)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
            
        x = self.norm(x)
        logits = self.output_head(x)
        return logits.float()


def count_params(model: nn.Module) -> int:
    """Conta i parametri addestrabili totali del modello."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(config: dict, seq_len: int) -> tuple[int, int]:
    """Calcola i FLOPs forward per token e quelli di training (3x) per token."""
    V = config["vocab_size"]
    d = config["d_model"]
    L = seq_len
    n_layers = config["num_layers"]
    n_q = config["num_heads"]
    
    n_kv = config.get("num_kv_heads")
    if n_kv is None:
        n_kv = n_q
        
    d_ff = config["d_ff"]
    
    hd = d // n_q
    attn_params = (d * d) + (2 * n_kv * hd * d) + (d * d)
    block_params = attn_params + (3 * d * d_ff)
    
    matmul_params = (n_layers * block_params) + (V * d)
    attn_score_flops = 4 * n_layers * L * d
    
    fwd_flops_per_token = (2 * matmul_params) + attn_score_flops
    train_flops_per_token = 3 * fwd_flops_per_token
    
    return fwd_flops_per_token, train_flops_per_token


# =====================================================================
# 4. TRAINING UTILITIES E GENERAZIONE (Sezione 4.2)
# =====================================================================

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood calcolata con il trucco log-sum-exp in fp32."""
    logits = logits.view(-1, logits.size(-1)).float()
    targets = targets.view(-1)
    
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - logits_max
    
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    log_probs = shifted_logits - log_sum_exp
    
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    return -target_log_probs.mean()


def clip_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """Scala i gradienti senza forzare sincronizzazioni CPU-GPU."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    # Calcola la norma totale (rimane un tensore sulla GPU)
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
    # Calcola il coefficiente interamente come tensore
    clip_coef = max_norm / (total_norm + eps)
    # Invece di usare un if (che bloccherebbe la GPU), usiamo torch.clamp
    # Se clip_coef è > 1.0, viene limitato a 1.0 (cioè i gradienti non vengono toccati)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in params:
        p.grad.detach().mul_(clip_coef_clamped)
            
    return total_norm


class AdamW(torch.optim.Optimizer):
    """Implementazione di AdamW con Decoupled Weight Decay."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                step = state["step"]

                p.mul_(1 - group["lr"] * group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def lr_cosine(step: int, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step <= total_steps:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    return lr_min

def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Campiona finestre casuali per l'addestramento in modo vettorizzato."""
    max_start_idx = len(data) - context_length - 1
    starts = np.random.randint(0, max_start_idx + 1, size=(batch_size, 1))
    # Creiamo gli offset per l'intera riga (1 x context_length)
    offsets = np.arange(context_length)
    # Sfruttiamo il broadcasting di NumPy per ottenere le matrici complete di indici
    x_indices = starts + offsets
    y_indices = starts + offsets + 1
    # Estraiamo i dati tutti in una volta e li convertiamo in tensori
    x = torch.from_numpy(data[x_indices].astype(np.int64)).to(device)
    y = torch.from_numpy(data[y_indices].astype(np.int64)).to(device)
    
    return x, y


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str) -> None:
    """Salva il modello, l'ottimizzatore e lo step corrente."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Carica il checkpoint e ritorna lo step a cui riprendere."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"]


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float, top_p: float, eos_id: int | None, seed: int) -> list[int]:
    """Generazione testo con temperature e nucleus (top-p) sampling."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    new_ids = []
    for _ in range(max_new_tokens):
        ctx_ids = ids[:, -model.context_length:]
        logits = model(ctx_ids)
        next_token_logits = logits[0, -1, :]
        
        next_token_logits = next_token_logits / temperature
        probs = softmax(next_token_logits, dim=-1)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            probs = softmax(next_token_logits, dim=-1)
            
        next_token = torch.multinomial(probs, num_samples=1, generator=generator).item()
        new_ids.append(next_token)
        
        if eos_id is not None and next_token == eos_id:
            break
            
        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)
        
    model.train()
    return new_ids
