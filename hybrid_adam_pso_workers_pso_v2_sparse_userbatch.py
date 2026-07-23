import os
import copy
import time
import random
import traceback
from queue import Empty
from multiprocessing import get_context
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


DEVICE = torch.device("cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SPLITS_PATH = "splits.pt"
BATCH_SIZE = 2048
MAX_SAMPLES_PER_SPLIT = None

EMB_DIM = 16
HIDDEN_DIM = 16
DROPOUT = 0.1
DROPOUT_EMB = 0.2

NUM_ROUNDS = 5
NUM_WORKERS = 5
GLOBAL_ADAM_EPOCHS = 4
BASE_LR = 1e-3
NUM_ADAM_TRAIN_SPLITS = 2  # None = use all splits; int = random n splits per round
MAX_ITEMS_PER_BATCH = 64

PSO_PARTICLES = 5
PSO_ITERS = 3
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5
PSO_NOISE_SCALE = 0.01

AGG_EPSILON = 1e-8

QUEUE_TIMEOUT = None
WORKER_JOIN_TIMEOUT = 30
TIME_BUDGET_SECONDS = 100000000000000000000000000000000


#Data loading

def load_tensor_splits(max_samples_per_split: int | None = MAX_SAMPLES_PER_SPLIT):
    assert os.path.exists(SPLITS_PATH), f"{SPLITS_PATH} not found. Run the data-split notebook first."

    raw_splits = torch.load(SPLITS_PATH)
    print(f"[Main] Loaded {len(raw_splits)} tensor splits from {SPLITS_PATH}")

    all_X_full = torch.cat(
        [torch.cat([s['X_train'], s['X_test']], dim=0) for s in raw_splits],
        dim=0,
    )
    n_users_global = int(all_X_full[:, 0].max().item()) + 1
    n_movies_global = int(all_X_full[:, 1].max().item()) + 1
    print(f"[Main] Global id ranges (full data): n_users_global={n_users_global}, n_movies_global={n_movies_global}")

    tensor_splits = []
    for i, s in enumerate(raw_splits):
        X_train, y_train = s["X_train"], s["y_train"]
        X_test, y_test = s["X_test"], s["y_test"]

        if max_samples_per_split is not None and X_train.size(0) > max_samples_per_split:
            X_train = X_train[:max_samples_per_split].clone()
            y_train = y_train[:max_samples_per_split].clone()
        if max_samples_per_split is not None and X_test.size(0) > max_samples_per_split:
            X_test = X_test[:max_samples_per_split].clone()
            y_test = y_test[:max_samples_per_split].clone()

        tensor_splits.append(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
        )

        print(
            f"[Main] Split {i}: "
            f"train={X_train.size(0)}, test={X_test.size(0)}"
        )

    return tensor_splits, n_users_global, n_movies_global


#Model

class CollabFiltering(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=EMB_DIM, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim, sparse=True)
        self.movie_emb = nn.Embedding(n_movies, emb_dim, sparse=True)
        self.dropout_emb = DROPOUT_EMB

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.ReLU(),
        )

    def forward(self, user, movie):
        u = F.dropout(self.user_emb(user), p=self.dropout_emb, training=self.training)
        m = F.dropout(self.movie_emb(movie), p=self.dropout_emb, training=self.training)
        x = torch.cat([u, m], dim=1)
        return self.mlp(x).squeeze()


loss_fn = nn.MSELoss()


def create_model(n_users_global, n_movies_global):
    model = CollabFiltering(
        n_users_global, n_movies_global,
        emb_dim=EMB_DIM, hidden=HIDDEN_DIM, dropout=DROPOUT
    )
    return model.to(DEVICE)


def make_loaders_from_split(split_dict, batch_size=BATCH_SIZE):
    X_train, y_train = split_dict["X_train"], split_dict["y_train"]
    X_test, y_test = split_dict["X_test"], split_dict["y_test"]

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_combined_train_loader(tensor_splits, batch_size=BATCH_SIZE):
    """Build a single train DataLoader from all splits' train data."""
    X_all = torch.cat([s["X_train"] for s in tensor_splits], dim=0)
    y_all = torch.cat([s["y_train"] for s in tensor_splits], dim=0)
    ds = TensorDataset(X_all, y_all)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def make_combined_test_loader(tensor_splits, batch_size=BATCH_SIZE):
    """Build a single test DataLoader from all splits' test data."""
    X_all = torch.cat([s["X_test"] for s in tensor_splits], dim=0)
    y_all = torch.cat([s["y_test"] for s in tensor_splits], dim=0)
    ds = TensorDataset(X_all, y_all)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def build_user_batches(X_train: torch.Tensor, y_train: torch.Tensor, max_items_per_batch: int = MAX_ITEMS_PER_BATCH):
    user_data = defaultdict(list)
    for idx in range(len(X_train)):
        u = int(X_train[idx][0])
        i = int(X_train[idx][1])
        r = float(y_train[idx])
        user_data[u].append((i, r))

    user_batches = []
    for u, items in user_data.items():
        for start in range(0, len(items), max_items_per_batch):
            chunk = items[start:start + max_items_per_batch]
            users = [u] * len(chunk)
            items_list = [x[0] for x in chunk]
            ratings = [x[1] for x in chunk]
            user_batches.append((users, items_list, ratings))
    return user_batches


def train_one_epoch_user_batches(model, user_batches, opt_sparse, opt_dense, loss_fn):
    model.train()
    total_loss = 0.0
    total_batches = 0

    random.shuffle(user_batches)
    for users, items, ratings in user_batches:
        users = torch.tensor(users).long().to(DEVICE)
        items = torch.tensor(items).long().to(DEVICE)
        ratings = torch.tensor(ratings).float().to(DEVICE)

        opt_sparse.zero_grad(set_to_none=True)
        opt_dense.zero_grad(set_to_none=True)
        preds = model(users, items)
        loss = loss_fn(preds, ratings)
        loss.backward()
        opt_sparse.step()
        opt_dense.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.float().to(DEVICE)
        preds = model(X_batch[:, 0].long(), X_batch[:, 1].long())
        loss = loss_fn(preds, y_batch)
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def aggregate_models_cpu(weights, node_states):
    n_nodes = len(node_states)
    assert len(weights) == n_nodes

    agg_state = {}
    for key in node_states[0].keys():
        agg_param = torch.zeros_like(node_states[0][key])
        for i in range(n_nodes):
            agg_param += weights[i] * node_states[i][key]
        agg_state[key] = agg_param
    return agg_state


#PSO Helper

def reconstruct_mixed_state(global_state, best_alpha, noise_seed, noise_scale):
    """
    Reconstruct worker's best state from alpha + seed (avoids sending full model).
    mixed = alpha * (global + noise) + (1-alpha) * global = global + alpha * noise
    """
    torch.manual_seed(noise_seed)
    mixed = {}
    for k, v in global_state.items():
        noise = noise_scale * torch.randn_like(v)
        mixed[k] = v + best_alpha * noise
    return mixed


def pso_over_alpha(global_state, local_state, val_loader, model_template,
                   loss_fn, num_particles=PSO_PARTICLES, max_iters=PSO_ITERS,
                   w=PSO_W, c1=PSO_C1, c2=PSO_C2):
    """
    Tiny PSO over a single scalar alpha mixing global/local states.
    Returns best_alpha only (no full state) for minimal communication.
    """
    particles = [random.random() for _ in range(num_particles)]  # alphas in [0,1]
    velocities = [0.0 for _ in range(num_particles)]
    pbest_pos = particles.copy()
    pbest_scores = [float("inf") for _ in range(num_particles)]

    gbest_pos = None
    gbest_score = float("inf")

    def mix_states(alpha):
        a = float(min(1.0, max(0.0, alpha)))
        mixed = {}
        for k in global_state.keys():
            mixed[k] = a * local_state[k] + (1.0 - a) * global_state[k]
        return mixed

    for it in range(max_iters):
        for i in range(num_particles):
            alpha = particles[i]
            mixed_state = mix_states(alpha)

            model = copy.deepcopy(model_template)
            model.load_state_dict(mixed_state)
            score = evaluate_model(model, val_loader, loss_fn)

            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_pos[i] = alpha

            if score < gbest_score:
                gbest_score = score
                gbest_pos = alpha

        for i in range(num_particles):
            r1 = random.random()
            r2 = random.random()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_pos[i] - particles[i])
                + c2 * r2 * ((gbest_pos if gbest_pos is not None else particles[i]) - particles[i])
            )
            particles[i] = particles[i] + velocities[i]

        print(f"[Worker-PSO] Iter {it + 1}/{max_iters} | best_score={gbest_score:.6f}")

    return float(gbest_pos), float(gbest_score)



#PSO

def worker_pso_node(worker_id, split_dict, global_state_dict, n_users_global, n_movies_global,
                    result_queue, round_idx, pso_particles=PSO_PARTICLES, pso_iters=PSO_ITERS,
                    batch_size=BATCH_SIZE, noise_scale=PSO_NOISE_SCALE):
    """
    Worker process: receives global model (after Adam in main),
    builds a perturbed version, and runs a small PSO over alpha.
    """
    # Avoid thread oversubscription when using multiple processes
    torch.set_num_threads(1)

    print(f"[Worker {worker_id}] Starting worker process.")
    try:
        # Recreate model and loaders inside this process
        model = create_model(n_users_global, n_movies_global)
        train_loader, test_loader = make_loaders_from_split(split_dict, batch_size=batch_size)

        # Load global params
        model.load_state_dict(global_state_dict)

        # Evaluate the pure global model on this worker's split (test set - no leakage)
        local_test_loss = evaluate_model(model, test_loader, loss_fn)
        print(f"[Worker {worker_id}] Using global model only. Local test_loss={local_test_loss:.6f}")

        # Build perturbed state with reproducible seed (main reconstructs from alpha+seed)
        noise_seed = (SEED + round_idx * 1000 + worker_id) % (2**32)
        torch.manual_seed(noise_seed)
        global_state = copy.deepcopy(global_state_dict)
        local_state = copy.deepcopy(global_state_dict)
        with torch.no_grad():
            for k, v in local_state.items():
                noise = noise_scale * torch.randn_like(v)
                local_state[k] = v + noise

        # PSO uses TRAIN for validation
        val_loader = train_loader
        model_template = create_model(n_users_global, n_movies_global)

        print(f"[Worker {worker_id}] Starting PSO over alpha (particles={pso_particles}, iters={pso_iters})...")
        best_alpha, best_score = pso_over_alpha(
            global_state,
            local_state,
            val_loader,
            model_template,
            loss_fn,
            num_particles=pso_particles,
            max_iters=pso_iters,
        )
        print(f"[Worker {worker_id}] PSO done. best_alpha={best_alpha:.4f}, best_local_val_loss={best_score:.6f}")

        # Minimal payload: alpha + seed (main reconstructs; no full model)
        result = {
            "worker_id": worker_id,
            "ok": True,
            "best_alpha": float(best_alpha),
            "noise_seed": int(noise_seed),
            "local_test_loss": float(local_test_loss),
            "pso_best_val_loss": float(best_score),
        }
        result_queue.put(result)
        print(f"[Worker {worker_id}] Result sent to main. Exiting.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Worker {worker_id}] ERROR: {e}\n{tb}")
        # Still notify main so it doesn't hang
        result_queue.put(
            {
                "worker_id": worker_id,
                "ok": False,
                "error": str(e),
                "traceback": tb,
            }
        )


def run_hybrid_adam_pso_workers_pso(
    num_rounds=NUM_ROUNDS,
    num_workers=NUM_WORKERS,
    global_adam_epochs=GLOBAL_ADAM_EPOCHS,
    base_lr=BASE_LR,
    pso_particles=PSO_PARTICLES,
    pso_iters=PSO_ITERS,
    queue_timeout=QUEUE_TIMEOUT,
    max_samples_per_split=MAX_SAMPLES_PER_SPLIT,
    num_adam_train_splits=NUM_ADAM_TRAIN_SPLITS,
):
    """
    Run the hybrid training. With the default settings and subsampling,
    this is designed to complete within ~5 minutes on a typical CPU.
    """
    tensor_splits, n_users_global, n_movies_global = load_tensor_splits(
        max_samples_per_split=max_samples_per_split
    )
    assert num_workers <= len(tensor_splits)

    print("\n[Main] =========================================")
    print("[Main] Hybrid v2 (sparse+userbatch): SparseAdam on embeddings, user-grouped batches, PSO in workers")
    print(
        f"[Main] num_rounds={num_rounds}, num_workers={num_workers}, "
        f"global_adam_epochs={global_adam_epochs}, base_lr={base_lr}, "
        f"pso_particles={pso_particles}, pso_iters={pso_iters}, "
        f"max_samples_per_split={max_samples_per_split}, num_adam_train_splits={num_adam_train_splits}, "
        f"max_items_per_batch={MAX_ITEMS_PER_BATCH}"
    )
    print("[Main] =========================================\n")

    ctx = get_context("spawn")
    torch.set_num_threads(1)

    global_model = create_model(n_users_global, n_movies_global)
    global_test_loader = make_combined_test_loader(tensor_splits)  # test always on all splits
    print(f"[Main] Adam test: all {len(tensor_splits)} splits combined")

    history = {
        "round": [],
        "global_train_loss": [],
        "global_test_loss": [],
        "avg_worker_test_loss": [],
    }

    reads_per_round = []
    writes_per_round = []

    start_time = time.time()

    for r in range(num_rounds):
        print(f"\n[Main] ===== Communication Round {r} =====")

        # Check overall time budget
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET_SECONDS:
            print(f"[Main] Time budget nearly exceeded ({elapsed:.1f}s). Stopping further rounds.")
            break

        # Adam Global
        n_train = num_adam_train_splits if num_adam_train_splits is not None else len(tensor_splits)
        n_train = min(n_train, len(tensor_splits))
        split_indices = list(range(len(tensor_splits)))
        random.seed(SEED + r)  # reproducible per round
        random.shuffle(split_indices)
        adam_train_splits = [tensor_splits[i] for i in split_indices[:n_train]]
        global_train_loader = make_combined_train_loader(adam_train_splits)
        print(f"[Main] Adam train: random {n_train} splits (indices {sorted(split_indices[:n_train])})")

        # Build user batch
        X_all = torch.cat([s["X_train"] for s in adam_train_splits], dim=0)
        y_all = torch.cat([s["y_train"] for s in adam_train_splits], dim=0)
        user_batches = build_user_batches(X_all, y_all, max_items_per_batch=MAX_ITEMS_PER_BATCH)

        # SparseAdam 
        sparse_params = list(global_model.user_emb.parameters()) + list(global_model.movie_emb.parameters())
        dense_params = [p for n, p in global_model.named_parameters() if not n.startswith("user_emb") and not n.startswith("movie_emb")]
        opt_sparse = torch.optim.SparseAdam(sparse_params, lr=base_lr)
        opt_dense = torch.optim.Adam(dense_params, lr=base_lr)

        print(f"[Main] Running global Adam for {global_adam_epochs} epochs BEFORE PSO workers...")
        for ge in range(global_adam_epochs):
            train_loss = train_one_epoch_user_batches(global_model, user_batches, opt_sparse, opt_dense, loss_fn)
            test_loss = evaluate_model(global_model, global_test_loader, loss_fn)
            print(f"[Main] Global epoch {ge + 1}/{global_adam_epochs} | train_loss={train_loss:.6f} | test_loss={test_loss:.6f}")

        # Checkpoint Global
        global_state = copy.deepcopy(global_model.state_dict())

        result_queue = ctx.Queue()

        workers = []
        for wid in range(num_workers):
            split_dict = tensor_splits[wid]
            print(f"[Main] Spawning worker {wid} for round {r}.")
            p = ctx.Process(
                target=worker_pso_node,
                args=(
                    wid,
                    split_dict,
                    global_state,
                    n_users_global,
                    n_movies_global,
                    result_queue,
                    r,
                    pso_particles,
                    pso_iters,
                    BATCH_SIZE,
                    PSO_NOISE_SCALE,
                ),
            )
            p.start()
            workers.append(p)

        writes_this_round = num_workers  # one result expected per worker

        # Collect results with timeout
        worker_states = []
        worker_test_losses = []
        print("[Main] Waiting for worker results...")

        any_dead = False
        success_count = 0
        for _ in range(num_workers):
            try:
                msg = result_queue.get(timeout=queue_timeout)
            except Empty:
                print(f"[Main] WARNING: Timeout ({queue_timeout}s) waiting for worker result in round {r}.")
                break

            wid = msg["worker_id"]
            if not msg.get("ok", True):
                print(f"[Main] ERROR: Worker {wid} reported failure: {msg.get('error')}")
                print(msg.get("traceback", ""))
                any_dead = True
                continue

            worker_test_losses.append(msg["local_test_loss"])
            recon = reconstruct_mixed_state(
                global_state,
                msg["best_alpha"],
                msg["noise_seed"],
                PSO_NOISE_SCALE,
            )
            worker_states.append(recon)
            success_count += 1

            print(
                f"[Main] Received from worker {wid} | "
                f"alpha={msg['best_alpha']:.4f}, local_test_loss={msg['local_test_loss']:.6f}"
            )

        reads_this_round = success_count

        # Ensure workers are cleaned up
        for p in workers:
            p.join(timeout=WORKER_JOIN_TIMEOUT)
            if p.exitcode not in (0, None):
                any_dead = True
                print(f"[Main] ERROR: Worker process pid={p.pid} exited with exitcode={p.exitcode}")

        if any_dead and success_count < num_workers:
            print("[Main] One or more workers crashed before sending results. Aborting training early.")
            break

        if success_count == 0:
            print("[Main] No successful worker results, aborting training.")
            break

        print(f"[Main] Collected {success_count}/{num_workers} worker results.")

        # Aggregate PSO-tuned worker models with weights inversely proportional to local_test_loss
        losses_t = torch.tensor(worker_test_losses, dtype=torch.float32)
        inv_losses = 1.0 / (losses_t + AGG_EPSILON)
        weights_t = inv_losses / inv_losses.sum()
        weights = [float(w) for w in weights_t.tolist()]
        agg_state = aggregate_models_cpu(weights, worker_states)
        print(f"[Main] Aggregated worker models with weights={[round(w, 4) for w in weights]}")

        global_model.load_state_dict(agg_state)

        final_train_loss = evaluate_model(global_model, global_train_loader, loss_fn)
        final_test_loss = evaluate_model(global_model, global_test_loader, loss_fn)
        history["round"].append(r)
        history["global_train_loss"].append(float(final_train_loss))
        history["global_test_loss"].append(float(final_test_loss))
        history["avg_worker_test_loss"].append(float(sum(worker_test_losses) / max(len(worker_test_losses), 1)))

        reads_per_round.append(reads_this_round)
        writes_per_round.append(writes_this_round)

        print(
            f"[Main] Round {r} summary | global_train_loss={final_train_loss:.6f} | global_test_loss={final_test_loss:.6f}, "
            f"avg_worker_test_loss={(sum(worker_test_losses) / max(len(worker_test_losses), 1)):.6f}, "
            f"writes={writes_this_round}, reads={reads_this_round}"
        )

    total_writes = sum(writes_per_round)
    total_reads = sum(reads_per_round)

    n_params = sum(p.numel() for p in global_model.parameters())
    bytes_full_model = n_params * 4  # float32
    bytes_minimal_per_worker = 5 * 8  # worker_id, alpha, seed, local_test_loss, pso_best_val_loss
    comm_stats = {
        "num_rounds": len(history["round"]),
        "num_workers": num_workers,
        "reads_per_round": reads_per_round,
        "writes_per_round": writes_per_round,
        "total_queue_writes": total_writes,
        "total_queue_reads": total_reads,
        "payload_mode": "minimal",
        "bytes_per_worker_read": bytes_minimal_per_worker,
        "bytes_full_model": bytes_full_model,
    }

    elapsed_total = time.time() - start_time
    comm_stats["processing_time_seconds"] = elapsed_total
    print(f"\n[Main] Run complete in {elapsed_total:.1f} seconds. Communication statistics:")
    for k, v in comm_stats.items():
        print(f"[Main]   {k}: {v}")

    return global_model.state_dict(), history, comm_stats


def main():
    print(f"Using device: {DEVICE}")
    state_dict, history, comm_stats = run_hybrid_adam_pso_workers_pso()

    out_path = "hybrid_adam_pso_workers_pso_v2_sparse_userbatch_results.pt"
    torch.save(
        {
            "state_dict": state_dict,
            "history": history,
            "comm_stats": comm_stats,
        },
        out_path,
    )
    print(f"[Main] Saved results to {out_path}")

    # Comparison plot vs old run (if present)
    try:
        import matplotlib.pyplot as plt

        old_path = "hybrid_adam_pso_workers_pso_v2_results.pt"
        if os.path.exists(old_path):
            old = torch.load(old_path)
            old_hist = old.get("history", {})

            plt.figure(figsize=(8, 4))
            if "global_test_loss" in old_hist and len(old_hist["global_test_loss"]) > 0:
                plt.plot(old_hist["round"], old_hist["global_test_loss"], label="Old hybrid test")
            plt.plot(history["round"], history["global_test_loss"], label="Sparse+userbatch hybrid test")
            plt.xlabel("Round")
            plt.ylabel("Global test loss (MSE)")
            plt.title("Hybrid Adam+PSO: Old vs Sparse+User-batching")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_path = "hybrid_adam_pso_workers_pso_v2_sparse_userbatch_comparison.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            print(f"[Main] Saved comparison plot to {plot_path}")
        else:
            print(f"[Main] No old results found at {old_path}; skipping comparison plot.")
    except Exception as e:
        print(f"[Main] Plotting skipped due to error: {e}")


if __name__ == "__main__":
    main()

