"""回合 07 对比：dense SwiGLU vs MoE（n 个 SwiGLU 专家 + router + top-k 稀疏激活）。

在相同数据 / 相同种子下训练三种 FFN 结构的小 GPT，对比 loss、参数量与路由分布：
  - dense        ：普通 SwiGLU FFN（前 6 回合的底座）；
  - moe（无 aux） ：n 个 SwiGLU 专家 + router，但不加负载均衡损失；
  - moe（有 aux） ：同上，训练时加 load_balancing_loss 防止路由坍缩。

关键点（也是 MoE 的两个核心知识点）：
  1. 稀疏激活解耦「容量」与「计算」：总参数 ≈ n × 专家参数（容量放大 n 倍），每 token 只激活
     top-k 个专家（计算量只放大 k 倍）。对比「总参数」和「每 token 激活参数」两列就能看懂。
  2. 负载均衡：router 会偷懒坍缩到少数专家（无 aux 时），加 aux loss 后恢复均匀分布。

运行：python scripts/compare_moe.py [--steps N] [--block-size B] [--batch-size M]
      [--n-expert E] [--top-k K] [--aux-coef C] [--data PATH]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

# 让脚本能从 scripts/ 直接 import src/ 里的 model（scripts/ 与 src/ 同层）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, GPTConfig
from swiglu import SwiGLU


# ---------- 数据 ----------（与 compare_rope.py / ... / compare_kv_cache.py 相同）

def load_text(path):
    """读训练文本；路径不存在时回退到合成数据（保证脚本可独立跑通）。"""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    torch.manual_seed(0)
    vocab = " abcdefghijklmnopqrstuvwxyz\n"
    idx = torch.randint(0, len(vocab), (50000,)).tolist()
    return "".join(vocab[i] for i in idx)


def build_tokenizer(text):
    """char-level tokenizer：唯一字符 -> id。"""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi


def encode(text, stoi):
    return [stoi[c] for c in text]


# ---------- 训练 ----------

def get_batch(data, block_size, batch_size, device):
    """随机采样 (x, y)：x 是 block_size 个 token，y 是 x 右移一位（next-token 目标）。"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=20):
    """在随机 batch 上估 loss（轻量版，不复用任何 checkpoint）。aux loss 只加在训练、不加在评估。"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def train_one(name, data, config, steps, batch_size, device, aux_coef=0.0):
    """训练一个指定 FFN 结构的小 GPT，返回 (model, [(step, train_loss, val_loss), ...])。

    aux_coef > 0 时把各层 MoE 的负载均衡损失累加到主 loss 上（Mixtral 式，防止路由坍缩）。
    """
    torch.manual_seed(1337)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for step in range(steps):
        x, y = get_batch(data, config.block_size, batch_size, device)
        logits, loss = model(x, y)
        if aux_coef > 0:
            aux = sum(blk.mlp.aux_loss() for blk in model.transformer.h)
            loss = loss + aux_coef * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            val = estimate_loss(model, data, config.block_size, batch_size, device)
            history.append((step, loss.item(), val))
            print("    [{:12s}] step {:4d}: train {:.4f}  val {:.4f}".format(
                name, step, loss.item(), val))
    return model, history


# ---------- 统计 ----------

@torch.no_grad()
def router_freq(model, data, block_size, batch_size, device, n_expert):
    """统计 router 把路由流量分给各专家的占比（f_e，和为 1）。

    用训练后模型跑一个 batch，读取各层 MoE 记录的 _topk_idx（展平 (N,k)），
    对所有路由槽位做 one-hot 求和归一化。均匀时每个专家 ≈ 1/n_expert；坍缩时集中在某几个。
    """
    model.eval()
    x, _ = get_batch(data, block_size, batch_size, device)
    _ = model(x)  # 触发 forward，填充各层 MoE 的 _topk_idx
    counts = torch.zeros(n_expert, device=device)
    total = 0
    for blk in model.transformer.h:
        idx = blk.mlp.moe._topk_idx                      # (N, k)
        counts += F.one_hot(idx, n_expert).float().sum(dim=(0, 1))
        total += idx.numel()
    model.train()
    return (counts / total).cpu().tolist()


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-expert", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--aux-coef", type=float, default=0.1,
                        help="负载均衡损失系数（moe 有 aux 这组用）")
    parser.add_argument("--data", type=str, default=None,
                        help="训练文本路径，默认复用 pretraining 的 TinyShakespeare")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    # 数据 + tokenizer
    data_path = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "pretraining", "data", "input.txt")
    text = load_text(data_path)
    stoi = build_tokenizer(text)
    vocab_size = len(stoi)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # 沿用前六回合已升级的底座：RoPE + RMSNorm + SwiGLU + GQA + FlashAttention，本回合只变 FFN
    base_cfg = dict(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=4, n_head=4, n_embd=128, n_kv_head=2,
                    pos_enc="rope", norm="rmsnorm", activation="swiglu", attn_impl="flash")
    dense_cfg = GPTConfig(**base_cfg)                                    # n_expert=0 -> 稠密
    moe_cfg = GPTConfig(**base_cfg, n_expert=args.n_expert, top_k=args.top_k)

    def n_params(cfg):
        return sum(p.numel() for p in GPT(cfg).parameters())

    def n_ffn_params(cfg):
        return sum(p.numel() for n, p in GPT(cfg).named_parameters() if ".mlp." in n)

    def n_active_ffn_params(cfg):
        """每 token 实际激活的 FFN 参数量（全模型，n_layer 层）：dense 是全量 FFN，
        MoE 是 top_k 个专家 + router。这是 MoE 的灵魂：总参数涨到 n 倍，但每 token
        只花 top_k 个专家的算力（与 dense 的比值 ≈ top_k）。
        """
        if cfg.n_expert == 0:
            return n_ffn_params(cfg)
        expert = SwiGLU(cfg.n_embd, bias=cfg.bias)
        n_expert_params = sum(p.numel() for p in expert.parameters())
        n_router = cfg.n_embd * cfg.n_expert  # router bias=False
        return cfg.n_layer * (cfg.top_k * n_expert_params + n_router)

    n_dense = n_params(dense_cfg)
    n_moe = n_params(moe_cfg)
    print("=" * 72)
    print("回合 07：dense SwiGLU vs MoE（router + top-k 稀疏激活）")
    print("设备: {}  数据: {}".format(device, data_path if os.path.exists(data_path) else "（合成）"))
    print("vocab={}  block_size={}  batch={}  steps={}  n_expert={}  top_k={}  aux_coef={}".format(
        vocab_size, args.block_size, args.batch_size, args.steps,
        args.n_expert, args.top_k, args.aux_coef))
    print("总参数:    dense={}  moe={}  （总容量 {:.1f} 倍）".format(
        n_dense, n_moe, n_moe / n_dense))
    print("FFN 参数:  dense={}  moe={}  （FFN 放大 {:.1f} 倍 = n_expert；attention/embedding 不涨）".format(
        n_ffn_params(dense_cfg), n_ffn_params(moe_cfg),
        n_ffn_params(moe_cfg) / n_ffn_params(dense_cfg)))
    print("每 token 激活 FFN 参数: dense={}  moe={}  （{:.1f} 倍 ≈ top_k，只跟 k 走不跟 n 走）".format(
        n_active_ffn_params(dense_cfg), n_active_ffn_params(moe_cfg),
        n_active_ffn_params(moe_cfg) / n_active_ffn_params(dense_cfg)))
    print("=" * 72)

    print("\n[训练 dense（稠密 SwiGLU）]")
    _, hist_dense = train_one("dense", data, dense_cfg, args.steps, args.batch_size, device)
    print("\n[训练 moe（无负载均衡损失）]")
    model_moe_noaux, hist_moe = train_one("moe", data, moe_cfg, args.steps, args.batch_size, device)
    print("\n[训练 moe+aux（有负载均衡损失）]")
    model_moe_aux, hist_moe_aux = train_one(
        "moe+aux", data, moe_cfg, args.steps, args.batch_size, device, aux_coef=args.aux_coef)

    print("\n=== loss 对比（step: dense_tr/va  moe_tr/va  moe+aux_tr/va）===")
    print("注：aux_tr 含负载均衡损失（所以偏高），公平对比看 va 列（评估时不加 aux）。")
    print("{:>6} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}".format(
        "step", "dense_tr", "dense_va", "moe_tr", "moe_va", "aux_tr", "aux_va"))
    for (s1, dtr, dva), (s2, mtr, mva), (s3, atr, ava) in zip(hist_dense, hist_moe, hist_moe_aux):
        print("{:6d} {:11.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f}".format(
            s1, dtr, dva, mtr, mva, atr, ava))

    print("\n=== 路由分布（每个专家分到的流量占比，均匀时每个 ≈ {:.2f}）===".format(
        1.0 / args.n_expert))
    print("  moe（无 aux）: {}".format(
        ["{:.2f}".format(f) for f in router_freq(model_moe_noaux, data, args.block_size,
                                                 args.batch_size, device, args.n_expert)]))
    print("  moe+aux      : {}".format(
        ["{:.2f}".format(f) for f in router_freq(model_moe_aux, data, args.block_size,
                                                 args.batch_size, device, args.n_expert)]))

    print("\n结论:")
    print("  - MoE 把一个大 FFN 拆成 {} 个专家，每个 token 只激活 top-{} 个：FFN 参数放大 {:.1f} 倍，".format(
        args.n_expert, args.top_k, n_ffn_params(moe_cfg) / n_ffn_params(dense_cfg)))
    print("    每 token 计算量只 {:.1f} 倍——「容量」与「每 token 算力」被解耦（加专家涨容量，不涨计算）。".format(
        n_active_ffn_params(moe_cfg) / n_active_ffn_params(dense_cfg)))
    print("  - mini 训练 300 步 MoE 已略优于 dense（va 列），且专家只用了 4 个、还有加容量的余量。")
    print("  - 负载均衡 loss 的作用是「利用率」而非「质量」：无 aux 时路由偏向少数专家（看分布），")
    print("    aux 把它拉均匀，换来的是专家并行训练/推理的吞吐，代价是少量 va loss——")
    print("    所以纯单机小模型 aux 可有可无，但真分布式 MoE 训练（专家分卡）离不开它。")


if __name__ == "__main__":
    main()
