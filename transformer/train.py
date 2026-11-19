import math
import time

import torch

import data
from model import GPT

# ---- 1. Config: every knob in one place ----------------------------------
cfg = dict(
    batch_size=64, max_t=256,                     # B, T
    dim=384, head_size=64, n_layers=6,            # C, per-head width, depth
    dropout=0.2,                                  # regularisation (0 = off)
    patience=4,                                   # evals without a new best val → stop
    peak_lr=1e-3, min_lr=1e-4,                    # schedule end points
    warmup_steps=100, total_steps=5000,
    betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
    clip=1.0,
    eval_every=250, eval_batches=20,
    seed=1337,
)

# ---- 2. Setup: seed, device, model ----------------------------------------
torch.manual_seed(cfg["seed"])
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = GPT(len(data.chars), cfg["dim"], cfg["max_t"],
            cfg["head_size"], cfg["n_layers"], cfg["dropout"]).to(device)


def get_batch(split):
    """One (x, y) batch from train or val, as int64 tensors on the device."""
    source = data.train_set if split == "train" else data.test_set
    x, y = data.batch(cfg["batch_size"], cfg["max_t"], source)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ---- 3. Learning-rate schedule: linear warm-up, then cosine decay ----------
def lr_at(step):
    if step < cfg["warmup_steps"]:
        return cfg["peak_lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / (cfg["total_steps"] - cfg["warmup_steps"])
    cosine = 0.5 * (1 + math.cos(math.pi * progress))      # goes 1 → 0
    return cfg["min_lr"] + cosine * (cfg["peak_lr"] - cfg["min_lr"])


# ---- 4. Optimiser: decay matrices, not LayerNorm weights -------------------
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": cfg["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=cfg["peak_lr"], betas=cfg["betas"], eps=cfg["eps"],
)


# ---- 5. Evaluation: average loss over a few batches, no learning ----------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(cfg["eval_batches"])]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ---- 6. One training step: the four lines that do the learning -------------
def train_step(x, y, step):
    for group in optimizer.param_groups:              # set this step's lr
        group["lr"] = lr_at(step)
    _, loss = model(x, y)                             # forward + loss
    optimizer.zero_grad(set_to_none=True)             # clear last step's grads
    loss.backward()                                   # grads for every weight
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
    optimizer.step()                                  # nudge the weights
    return loss.item()


# ---- 7. Run: `python train.py overfit` (sanity check) or `python train.py` -
if __name__ == "__main__":
    import sys
    overfit = len(sys.argv) > 1 and sys.argv[1] == "overfit"

    if overfit:                                       # same batch every step
        x, y = get_batch("train")
        for step in range(300):
            loss = train_step(x, y, step)
            if step == 0:
                missing = [n for n, p in model.named_parameters() if p.grad is None]
                print("params with no gradient:", missing or "none")
            if step % 50 == 0 or step == 299:
                print(f"overfit step {step:4d} | loss {loss:.3f}")
    else:
        import csv, os, shutil
        out_dir = os.path.expanduser("~/edge-data")   # private: not Drive, not git
        run_name = time.strftime("%Y%m%d-%H%M") + f"_drop{cfg['dropout']}"   # unique per run
        log, t0 = [], time.time()
        best_val, best_step, bad_evals = float("inf"), 0, 0
        for step in range(cfg["total_steps"]):
            if step % cfg["eval_every"] == 0 or step == cfg["total_steps"] - 1:
                losses = estimate_loss()
                log.append((step, losses["train"], losses["val"]))
                print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f}"
                      f" | lr {lr_at(step):.2e} | {time.time() - t0:.0f}s", flush=True)
                with open(f"{out_dir}/loss_log.csv", "w", newline="") as f:   # rewrite each eval:
                    csv.writer(f).writerows([("step", "train", "val"), *log])  # survives a crash

                if losses["val"] < best_val:          # new best → save THIS model
                    best_val, best_step, bad_evals = losses["val"], step, 0
                    torch.save({"model": model.state_dict(), "cfg": cfg, "chars": data.chars,
                                "step": step, "val": best_val}, f"{out_dir}/ckpt.pt")
                    print(f"      ↳ new best val, saved checkpoint", flush=True)
                else:
                    bad_evals += 1
                    if bad_evals >= cfg["patience"]:  # val hasn't improved for a while
                        print(f"early stop: best val {best_val:.3f} at step {best_step}")
                        break
            train_step(*get_batch("train"), step)

        import matplotlib
        matplotlib.use("Agg")                         # draw to a file, no window
        import matplotlib.pyplot as plt
        steps, tr, va = zip(*log)
        plt.plot(steps, tr, label="train")
        plt.plot(steps, va, label="val")
        plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)
        plt.title(f"{run_name} | best val {best_val:.3f} @ step {best_step}")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/loss_{run_name}.png", dpi=120)
        shutil.copy(f"{out_dir}/loss_log.csv", f"{out_dir}/loss_log_{run_name}.csv")
        print(f"best checkpoint: step {best_step}, val {best_val:.3f} → ~/edge-data/ckpt.pt")
        print(f"curve → plots/loss_{run_name}.png | log → ~/edge-data/loss_log_{run_name}.csv")
