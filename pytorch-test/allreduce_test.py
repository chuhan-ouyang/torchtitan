import os, time, datetime, socket, argparse
import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--size-mb", type=int, default=128)
p.add_argument("--iters", type=int, default=5)
args = p.parse_args()

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local)

# newer torch supports device_id=...
kwargs = {"timeout": datetime.timedelta(seconds=300)}
try:
    dist.init_process_group("nccl", device_id=local, **kwargs)
except TypeError:
    dist.init_process_group("nccl", **kwargs)  # older torch fallback

dist.barrier()

# ---- correctness pass (single op) ----
x = torch.ones(args.size_mb * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
ok = torch.allclose(x, torch.full_like(x, float(world)))

# ---- timed pass (separate buffer; no correctness drift) ----
y = torch.ones_like(x)
# warmup
for _ in range(2):
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
dist.barrier()

# reset y to 1s so each iter ends at "world" (not compounding)
y.fill_(1.0)
torch.cuda.synchronize()
dist.barrier()

t0 = time.time()
for _ in range(args.iters):
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
t1 = time.time()

if rank == 0:
    dur = (t1 - t0) / args.iters
    sz_bytes = x.numel() * 4
    alg_bw_gbps = (sz_bytes / dur) / 1e9
    print("--------")
    print(
        f"host={socket.gethostname()} world={world} size={args.size_mb}MB "
        f"iters={args.iters} time/op={dur*1000:.2f} ms alg_bw~{alg_bw_gbps:.2f} GB/s "
        f"nccl={torch.cuda.nccl.version()} ok={bool(ok)}",
        flush=True,
    )
    print("--------")

dist.destroy_process_group()
