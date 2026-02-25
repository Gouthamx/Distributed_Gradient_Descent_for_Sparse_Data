# Federated-style CPU training script using threads + MSELoss + RMSE = sqrt(MSE)
import time
import math
import argparse
import pickle
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class CollabFiltering(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=16, hidden=16, dropout=0.1):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.movie_emb = nn.Embedding(n_movies, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.ReLU()
        )
    
    def forward(self, u_idx, m_idx):
        u = F.dropout(self.user_emb(u_idx), p=0.4, training=self.training)
        m = F.dropout(self.movie_emb(m_idx), p=0.4, training=self.training)
        return self.mlp(torch.cat([u, m], dim=1)).squeeze()


def weighted_average_state_dicts(state_dicts, weights):
    """Weighted average of state dicts based on test losses (lower loss = higher weight)"""
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / total_weight for w in weights]
    
    avg = {}
    for k in state_dicts[0].keys():
        stacked = torch.stack([sd[k].cpu() for sd in state_dicts], dim=0)
        avg[k] = sum(w * stacked[i] for i, w in enumerate(weights))
    return avg


def worker_proc(worker_id, tensors_dict, shared, shared_lock, cfg, global_n_users, global_n_movies):
    torch.manual_seed(cfg.seed + worker_id)

    X_train, y_train = tensors_dict["X_train"], tensors_dict["y_train"].float()
    X_test, y_test = tensors_dict["X_test"], tensors_dict["y_test"].float()
    
    model = CollabFiltering(global_n_users, global_n_movies, cfg.emb_dim, cfg.hidden, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    
    tr_loader = DataLoader(
        TensorDataset(X_train[:, 0].long(), X_train[:, 1].long(), y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0
    )
    te_loader = DataLoader(
        TensorDataset(X_test[:, 0].long(), X_test[:, 1].long(), y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load global state if available
    with shared_lock:
        global_state = shared.get("global_state", None)
    if global_state is not None:
        try:
            model.load_state_dict(global_state)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        except Exception as e:
            print(f"[W{worker_id}] failed to load initial global_state: {e}")
    
    history = []
    
    for round_idx in range(cfg.comm_rounds):
        best_rmse, best_test_loss, best_state = math.inf, math.inf, None
        
        for epoch in range(cfg.local_epochs):
            # Training
            model.train()
            losses = []
            for u, m, y in tr_loader:
                optimizer.zero_grad()
                preds = model(u, m)
                # Ensure same shape for loss calculation
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                y = y.view_as(preds)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            train_loss = sum(losses) / len(losses) if losses else 0.0
            train_rmse = math.sqrt(train_loss) if train_loss >= 0 else float("nan")
            
            # Validation
            model.eval()
            losses = []
            with torch.no_grad():
                for u, m, y in te_loader:
                    preds = model(u, m)
                    if preds.dim() == 0:
                        preds = preds.unsqueeze(0)
                    y = y.view_as(preds)
                    losses.append(loss_fn(preds, y).item())
            test_loss = sum(losses) / len(losses) if losses else 0.0
            test_rmse = math.sqrt(test_loss) if test_loss >= 0 else float("nan")
            
            history.append({
                "round": round_idx,
                "epoch_in_round": epoch,
                "global_epoch": round_idx * cfg.local_epochs + epoch,
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "test_loss": test_loss,
                "test_rmse": test_rmse
            })
            
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_test_loss = test_loss
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            
            if (epoch + 1) % cfg.log_every == 0:
                print(
                    f"[W{worker_id}] R{round_idx} E{epoch+1}/{cfg.local_epochs} "
                    f"train_loss={train_loss:.4f} train_rmse={train_rmse:.4f} "
                    f"test_loss={test_loss:.4f} test_rmse={test_rmse:.4f}"
                )
        
        # Signal this worker's best for this round
        with shared_lock:
            shared[f"w{worker_id}_r{round_idx}_best"] = best_state
            shared[f"w{worker_id}_r{round_idx}_rmse"] = best_rmse
            shared[f"w{worker_id}_r{round_idx}_test_loss"] = best_test_loss
            shared[f"w{worker_id}_r{round_idx}_ready"] = True
        
        # Wait for aggregated model
        agg_key = f"agg_r{round_idx}_state"
        waited = 0.0
        while waited <= cfg.agg_timeout:
            with shared_lock:
                agg_state = shared.get(agg_key, None)
            if agg_state is not None:
                try:
                    model.load_state_dict(agg_state)
                    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
                    print(f"[W{worker_id}] loaded aggregated model for round {round_idx}")
                except Exception as e:
                    print(f"[W{worker_id}] failed to load agg: {e}")
                break
            time.sleep(0.2)
            waited += 0.2
        else:
            print(f"[W{worker_id}] waited {cfg.agg_timeout}s for agg; continuing without it")
    
    with shared_lock:
        shared[f"w{worker_id}_final"] = history
        shared[f"w{worker_id}_done"] = True
    print(f"[W{worker_id}] finished all rounds")


def aggregator(num_workers, cfg, shared, shared_lock):
    for r in range(cfg.comm_rounds):
        print(f"[AGG] waiting for {num_workers} workers ready for round {r}...")
        
        # Wait until all workers mark this round as ready
        while True:
            with shared_lock:
                ready_count = sum(
                    1 for w in range(num_workers)
                    if shared.get(f"w{w}_r{r}_ready", False)
                )
            if ready_count >= num_workers:
                break
            time.sleep(0.2)
        
        with shared_lock:
            states = [shared[f"w{w}_r{r}_best"] for w in range(num_workers)]
            test_losses = [shared[f"w{w}_r{r}_test_loss"] for w in range(num_workers)]
            rmses = [shared[f"w{w}_r{r}_rmse"] for w in range(num_workers)]
        
        # Weighted average based on test loss (inverse weights: lower loss = higher weight)
        weights = [1.0 / (tl + 1e-8) for tl in test_losses]
        agg_state = weighted_average_state_dicts(states, weights)
        
        with shared_lock:
            shared[f"agg_r{r}_state"] = agg_state
        
        print(
            f"[AGG] published aggregated state for round {r} (weighted by test loss), "
            f"worker test_losses={test_losses}, rmses={rmses}"
        )
        
        # Clear ready flags for this round
        with shared_lock:
            for w in range(num_workers):
                rk = f"w{w}_r{r}_ready"
                if rk in shared:
                    del shared[rk]
    
    # Wait for all workers to finish
    print("[AGG] Waiting for all workers to finish...")
    while True:
        with shared_lock:
            done_count = sum(
                1 for w in range(num_workers)
                if shared.get(f"w{w}_done", False)
            )
        if done_count >= num_workers:
            break
        time.sleep(0.2)
    
    print("[AGG] Collecting worker histories...")
    all_candidates = []
    with shared_lock:
        for w in range(num_workers):
            history = shared.get(f"w{w}_final", [])
            print(f"[AGG] Worker {w} history length: {len(history)}")
            if history:
                all_candidates.append({
                    "node": w,
                    "epoch_states": None,
                    "train_losses": [e["train_loss"] for e in history],
                    "test_losses": [e["test_loss"] for e in history],
                    "train_rmses": [e["train_rmse"] for e in history],
                    "test_rmses": [e["test_rmse"] for e in history],
                })
            else:
                print(f"[AGG] WARNING: Worker {w} has no history!")
        
        shared["all_candidates_stage1"] = all_candidates
        shared["agg_done"] = True
    
    print(f"[AGG] Collected histories from {len(all_candidates)} workers")


def save_results(shared, shared_lock, cfg, num_workers, filename="ddp_results.pkl"):
    """Save results to file for plotting in notebook"""
    # Wait for aggregator to finish
    print("[SAVE] Waiting for aggregator to finish...")
    while True:
        with shared_lock:
            if shared.get("agg_done", False):
                break
        time.sleep(0.5)
    
    print("[SAVE] Waiting for all workers to finish...")
    while True:
        with shared_lock:
            done_count = sum(
                1 for w in range(num_workers)
                if shared.get(f"w{w}_done", False)
            )
        if done_count >= num_workers:
            break
        time.sleep(0.5)
        print(f"[SAVE] {done_count}/{num_workers} workers done...")
    
    with shared_lock:
        all_candidates = shared.get("all_candidates_stage1", [])
        if not all_candidates:
            print("[SAVE] No results in all_candidates_stage1, checking worker histories directly...")
            all_candidates = []
            for w in range(num_workers):
                history = shared.get(f"w{w}_final", [])
                print(f"[SAVE] Worker {w} history length: {len(history)}")
                if history:
                    all_candidates.append({
                        "node": w,
                        "epoch_states": None,
                        "train_losses": [e["train_loss"] for e in history],
                        "test_losses": [e["test_loss"] for e in history],
                        "train_rmses": [e["train_rmse"] for e in history],
                        "test_rmses": [e["test_rmse"] for e in history],
                    })
    
    if not all_candidates:
        print("[SAVE] No results to save - all_candidates is empty!")
        return
    
    print(f"[SAVE] Saving {len(all_candidates)} candidates")
    for i, cand in enumerate(all_candidates[:2]):
        print(
            f"[SAVE] Candidate {i}: "
            f"train_losses={len(cand.get('train_losses', []))}, "
            f"test_losses={len(cand.get('test_losses', []))}"
        )
    
    results = {
        "all_candidates": all_candidates,
        "comm_rounds": cfg.comm_rounds,
        "local_epochs": cfg.local_epochs,
        "num_workers": num_workers,
    }
    
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"[SAVE] Results saved to {filename}")


def main(cfg):
    tensors_list = torch.load("splits.pt", weights_only=False)
    if not isinstance(tensors_list, list) or len(tensors_list) == 0:
        raise RuntimeError("splits.pt not found or invalid. Run the notebook cell to create splits.pt")
    
    num_workers = min(cfg.num_workers, len(tensors_list))
    
    # Shared dict between threads with lock for thread safety
    shared = {}
    shared["global_state"] = None
    shared["agg_done"] = False
    shared_lock = threading.Lock()
    
    global_n_users = max(
        max(int(t["X_train"][:, 0].max().item()), int(t["X_test"][:, 0].max().item()))
        for t in tensors_list
    ) + 1
    global_n_movies = max(
        max(int(t["X_train"][:, 1].max().item()), int(t["X_test"][:, 1].max().item()))
        for t in tensors_list
    ) + 1
    
    # Spawn worker threads
    threads = [
        threading.Thread(
            target=worker_proc,
            args=(w, tensors_list[w], shared, shared_lock, cfg, global_n_users, global_n_movies),
            daemon=False,
        )
        for w in range(num_workers)
    ]
    
    for t in threads:
        t.start()
    
    # Aggregator runs in a separate thread
    agg_thread = threading.Thread(
        target=aggregator,
        args=(num_workers, cfg, shared, shared_lock),
        daemon=False
    )
    agg_thread.start()
    
    # Wait for aggregator to finish
    agg_thread.join()
    
    # Wait for all worker threads to finish
    for t in threads:
        t.join()
    
    # Save results
    save_results(shared, shared_lock, cfg, num_workers, filename="ddp_thread_results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--comm_rounds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=2)
    parser.add_argument("--agg_timeout", type=float, default=30.0)
    parser.add_argument("--drop_last", action="store_true")
    main(parser.parse_args())
