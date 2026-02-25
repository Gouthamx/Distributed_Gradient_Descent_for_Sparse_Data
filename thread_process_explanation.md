# How Threads and Processes Work in Your Code

## THREADING APPROACH (`ddp_thread.py`)

### 1. Initialization and Setup
- **Shared Memory Creation:**
  - `shared = {}` - Regular Python dictionary created in main process
  - `shared_lock = threading.Lock()` - Lock object for thread synchronization
  - All threads share the **same memory address** for the dictionary
  - No copying or serialization needed

- **Global Dimensions:**
  - Calculates `global_n_users` and `global_n_movies` from all data splits
  - Used to create models with consistent embedding dimensions across workers

### 2. Thread Creation
```python
threads = [
    threading.Thread(
        target=worker_proc,
        args=(w, tensors_list[w], shared, shared_lock, cfg, global_n_users, global_n_movies),
        daemon=False,
    )
    for w in range(num_workers)
]
```
- **Creates N worker threads** (default: 5)
- Each thread runs `worker_proc()` function
- Passes: worker_id, data split, shared dict, lock, config, global dimensions
- `daemon=False` means threads must complete before main process exits

### 3. Thread Execution Flow
- **Start Threads:**
  - `t.start()` - Starts each thread (non-blocking)
  - All threads begin execution immediately
  - **BUT:** GIL ensures only one executes Python bytecode at a time

- **Aggregator Thread:**
  - Separate thread runs `aggregator()` function
  - Monitors shared dict for worker completion signals
  - Performs model aggregation when all workers ready

### 4. Worker Thread Operations (`worker_proc`)

**Training Phase:**
- Each thread trains on its assigned data split
- Creates local model: `CollabFiltering(global_n_users, global_n_movies, ...)`
- Trains for `local_epochs` (default: 10) per communication round
- Tracks best model state based on test RMSE

**Communication Phase (Per Round):**
```python
with shared_lock:
    shared[f"w{worker_id}_r{round_idx}_best"] = best_state
    shared[f"w{worker_id}_r{round_idx}_rmse"] = best_rmse
    shared[f"w{worker_id}_r{round_idx}_test_loss"] = best_test_loss
    shared[f"w{worker_id}_r{round_idx}_ready"] = True
```
- **Write Operations:**
  - Acquires lock (blocks other threads)
  - Directly writes to shared dict (same memory location)
  - Releases lock
  - **Fast:** ~0.001ms per operation (direct memory access)

**Waiting for Aggregated Model:**
```python
while waited <= cfg.agg_timeout:
    with shared_lock:
        agg_state = shared.get(agg_key, None)
    if agg_state is not None:
        model.load_state_dict(agg_state)
        break
    time.sleep(0.2)
```
- **Read Operations:**
  - Polls shared dict every 0.2 seconds
  - Acquires lock, checks for aggregated state
  - Releases lock, sleeps if not available
  - Loads aggregated model when available

### 5. Aggregator Thread Operations (`aggregator`)

**Waiting for Workers:**
```python
while True:
    with shared_lock:
        ready_count = sum(1 for w in range(num_workers) 
                         if shared.get(f"w{w}_r{r}_ready", False))
    if ready_count >= num_workers:
        break
    time.sleep(0.2)
```
- Polls shared dict to check if all workers completed round
- Uses lock to safely read ready flags

**Reading Worker States:**
```python
with shared_lock:
    states = [shared[f"w{w}_r{r}_best"] for w in range(num_workers)]
    test_losses = [shared[f"w{w}_r{r}_test_loss"] for w in range(num_workers)]
```
- Acquires lock
- Reads all worker model states (direct memory access)
- Releases lock

**Writing Aggregated State:**
```python
with shared_lock:
    shared[f"agg_r{r}_state"] = agg_state
```
- Computes weighted average of all worker states
- Writes aggregated model back to shared dict
- Workers can now read and load it

### 6. Thread Synchronization
- **Lock Mechanism:**
  - `threading.Lock()` ensures only one thread accesses shared dict at a time
  - Prevents race conditions (concurrent read/write)
  - Threads block (wait) when lock is held by another thread

- **GIL Limitation:**
  - Global Interpreter Lock prevents true parallel execution
  - Only one thread executes Python bytecode at a time
  - Threads take turns (time-slicing), not truly parallel
  - CPU-bound training runs **sequentially**, not in parallel

### 7. Completion and Cleanup
- **Wait for Completion:**
  - `agg_thread.join()` - Main thread waits for aggregator to finish
  - `t.join()` - Main thread waits for each worker thread to finish
  - Ensures all threads complete before saving results

- **Result Saving:**
  - Reads final histories from shared dict
  - Saves to `ddp_thread_results.pkl`

---

## MULTIPROCESSING APPROACH (`ddp.py`)

### 1. Initialization and Setup
- **Manager Process Creation:**
  ```python
  shared = Manager().dict()
  ```
  - `Manager()` spawns a **separate manager process**
  - Manager process maintains the actual dictionary
  - Returns a **proxy object** (not the real dict)
  - Proxy forwards operations to manager via IPC (Inter-Process Communication)

- **Start Method:**
  ```python
  set_start_method("spawn")
  ```
  - Windows-compatible process creation method
  - Each process starts fresh (no shared memory)

### 2. Process Creation
```python
procs = [Process(target=worker_proc, 
                 args=(w, tensors_list[w], shared, cfg, global_n_users, global_n_movies))
         for w in range(num_workers)]
```
- **Creates N worker processes** (default: 5)
- Each process runs `worker_proc()` function
- Passes: worker_id, data split, **proxy object**, config, global dimensions
- Each process has **separate memory space**

### 3. Process Execution Flow
- **Start Processes:**
  - `p.start()` - Starts each process (non-blocking)
  - Each process runs in **separate Python interpreter**
  - **True parallel execution** (bypasses GIL)
  - Processes can run simultaneously on different CPU cores

- **Aggregator:**
  - Runs in **main process** (not separate process/thread)
  - Can run while worker processes train in parallel
  - Accesses shared dict through proxy

### 4. Worker Process Operations (`worker_proc`)

**Training Phase:**
- Each process trains on its assigned data split
- Creates local model in **separate memory space**
- Trains for `local_epochs` per communication round
- **Runs in parallel** with other processes (true parallelism)

**Communication Phase (Per Round):**
```python
shared[f"w{worker_id}_r{round_idx}_best"] = best_state
shared[f"w{worker_id}_r{round_idx}_rmse"] = best_rmse
shared[f"w{worker_id}_r{round_idx}_test_loss"] = best_test_loss
shared[f"w{worker_id}_r{round_idx}_ready"] = True
```
- **Write Operations (via Proxy):**
  1. Proxy intercepts assignment
  2. **Serializes** (pickles) the state_dict → bytes
  3. Sends bytes through **pipe/socket** to Manager process
  4. Manager receives, **deserializes**, stores in its dict
  5. Manager sends acknowledgment back
  - **Slower:** ~10-50ms per operation (serialization + IPC overhead)
  - **No explicit lock needed** (Manager handles synchronization)

**Waiting for Aggregated Model:**
```python
while waited <= cfg.agg_timeout:
    if shared.get(agg_key):
        model.load_state_dict(shared[agg_key])
        break
    time.sleep(0.2)
```
- **Read Operations (via Proxy):**
  1. Proxy sends read request to Manager
  2. Manager reads from its dict
  3. Manager **serializes** the state_dict → bytes
  4. Sends bytes back through pipe
  5. Worker receives, **deserializes**, gets copy
  - Creates **new copy** in worker's memory (not same object)

### 5. Aggregator Operations (`aggregator`)

**Waiting for Workers:**
```python
while sum(1 for w in range(num_workers) 
          if shared.get(f"w{w}_r{r}_ready", False)) < num_workers:
    time.sleep(0.2)
```
- Polls shared dict (via proxy) to check worker completion
- **No explicit lock** - Manager handles synchronization internally

**Reading Worker States:**
```python
states = [shared[f"w{w}_r{r}_best"] for w in range(num_workers)]
test_losses = [shared[f"w{w}_r{r}_test_loss"] for w in range(num_workers)]
```
- Each read goes through proxy → Manager → serialization → deserialization
- Gets **copies** of worker states (not same objects)

**Writing Aggregated State:**
```python
shared[f"agg_r{r}_state"] = weighted_average_state_dicts(states, weights)
```
- Computes weighted average
- Writes through proxy → Manager (serialization + IPC)
- Workers can now read aggregated model

### 6. Process Synchronization
- **Manager Process:**
  - All operations go through Manager process
  - Manager handles operations **sequentially** (one at a time)
  - Built-in synchronization (no explicit locks needed)
  - Each operation is atomic at Manager level

- **True Parallelism:**
  - Each process has separate Python interpreter
  - **No GIL** - processes run truly in parallel
  - Multiple CPU cores utilized simultaneously
  - Training happens **in parallel**, not sequentially

### 7. Completion and Cleanup
- **Wait for Completion:**
  - `aggregator()` runs in main process (blocks until done)
  - `p.join()` - Main process waits for each worker process to finish
  - Ensures all processes complete before saving

- **Result Saving:**
  - Reads final histories from shared dict (via proxy)
  - Saves to `ddp_process_results.pkl`

---

## KEY DIFFERENCES IN YOUR CODE

### Memory Access
| Aspect | Threading | Multiprocessing |
|--------|-----------|----------------|
| **Shared Dict** | `shared = {}` (regular dict) | `shared = Manager().dict()` (proxy) |
| **Memory Location** | Same memory address (all threads) | Separate memory (Manager process) |
| **Access Method** | Direct pointer access | Proxy → IPC → Manager |
| **Lock Required** | Yes (`shared_lock`) | No (Manager handles it) |

### Write Operation
**Threading:**
```python
with shared_lock:
    shared["key"] = value  # Direct memory write
```

**Multiprocessing:**
```python
shared["key"] = value  # Proxy → Serialize → IPC → Manager → Deserialize
```

### Read Operation
**Threading:**
```python
with shared_lock:
    value = shared["key"]  # Direct memory read (same object)
```

**Multiprocessing:**
```python
value = shared["key"]  # Proxy → Request → Manager → Serialize → IPC → Deserialize (new copy)
```

### Execution Model
**Threading:**
- All threads in same process
- GIL prevents parallel execution
- Sequential training (workers take turns)
- Fast communication, slow computation

**Multiprocessing:**
- Separate processes
- True parallel execution (no GIL)
- Parallel training (workers run simultaneously)
- Slow communication, fast computation

### Performance Impact
**Threading:**
- Communication: ~0.04ms per round (fast)
- Training: Sequential (5 workers × single time)
- **Total: Slow** (computation dominates, no speedup)

**Multiprocessing:**
- Communication: ~2000ms per round (slow)
- Training: Parallel (single time / 5 workers)
- **Total: Fast** (parallel computation >> communication overhead)

---

## SUMMARY

### Threading (`ddp_thread.py`):
- ✅ Fast communication (direct memory access)
- ✅ Simple synchronization (explicit locks)
- ❌ Sequential execution (GIL limitation)
- ❌ No speedup from multiple workers
- ❌ Wastes CPU cores

### Multiprocessing (`ddp.py`):
- ✅ True parallelism (bypasses GIL)
- ✅ Utilizes all CPU cores
- ✅ Significant speedup with multiple workers
- ❌ Slower communication (serialization overhead)
- ❌ Higher memory usage (copies in each process)

**For CPU-bound neural network training, multiprocessing is preferred despite slower communication because parallel execution provides much greater speedup than the communication overhead.**

