import os, torch
import torch.distributed as dist
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl  # noqa: registers hccl backend
rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="hccl", rank=rank, world_size=ws)
t = torch.ones(8, device="hpu") * (rank + 1)
dist.all_reduce(t)
htcore.mark_step()
exp = ws * (ws + 1) // 2
ok = abs(t[0].item() - exp) < 1e-3
if rank == 0:
    print(f"[HCCL] world={ws} allreduce_sum0={t[0].item()} expected={exp} -> {'OK' if ok else 'MISMATCH'}")
dist.destroy_process_group()
