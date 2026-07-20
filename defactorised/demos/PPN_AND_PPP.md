# PPn and PPP are different architectures

These CPU-safe demos make two repository execution patterns small enough to
read and run during a one-hour walkthrough. They are teaching programs, not
campaign launchers.

## PPn: forward wavefront

`ppn_stage_demo.py` represents a model split into contiguous stages. For each
tile, stage `s` consumes only the detached output of stage `s-1`, applies its
owned transform, and publishes to stage `s+1`. The dependency graph is:

```text
O[s,t] -> O[s+1,t]   downstream progress
O[s,t] -> O[s,t+1]   next tile on the same stage
```

`ppn_demo.sh` coordinates the workers and terminates siblings when a stage
fails. Its JSON packets make routing and atomic publication visible, but they
are only a CPU pedagogical stand-in. Production same-node handoff must be
RAM-backed under `/dev/shm`; cross-node production handoff is NCCL over
InfiniBand, never JSON traffic through `/tmp`, Lustre, NFS, or SSD.

`ppn_partition_demo.py` is a separate planning companion. It reads measured
per-block costs and chooses contiguous cuts that minimize the predicted p95
slowest stage. Production pins the resulting cuts and profile identity; it
does not silently repartition at launch.

## Pipeline-v4 PPP: independent block shards

`ppp_independent_stage_demo.py` represents the structurally independent v4
stage workers. Every worker owns disjoint blocks and performs only local
teacher-sourced updates. There are no inter-stage training activation
boundaries and no training wavefront. Its output is an independently
mergeable stage shard.

The production v4 relay supports coordinated store-fill and evaluation. That
relay does not turn independent shard training into PPn activation pipelining.
`ppp_demo.sh` coordinates the independent synthetic workers and collects the
shards.

In short: **PPn passes detached activations forward; PPP trains disjoint
shards independently.** Similar stage numbering does not make them synonyms.
