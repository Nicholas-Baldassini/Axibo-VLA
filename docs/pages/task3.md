# Task 3
Multiple optimizations were tested and compared together. Listed below. All configurable in the config [`train/config/speed_eval.py`](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/train/config/speed_eval.py) (which stacks to run, how many rollouts of each pair, lockstep or real time), run by [`train/speed_eval/speed_eval.py`](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/train/speed_eval/speed_eval.py), one file per optimization in [`train/speed_eval/`](https://github.com/Nicholas-Baldassini/AxiboVLAtest/tree/main/train/speed_eval).



## Lockstep

In task 2, the policy was evaluated in lockstep. So the sim would freeze as the policy makes a prediction. Lockstep approaches reality as inference latency approaches zero. 

## Test suite

These were the different optimizations tested
### Engine level optimizations
- **b16**: bf16 is the precision the policy was trained in and what every stack below runs. this option on top of vanilla additionally merges the LoRA adapter into the base weights (one matmul per layer instead of three), so "b16" below really means "merged"
- **b4**: lower precision, 4-bit weights, what a 12 GB card has to use
- **torch_compile**: the decode step compiled with a static KV cache and CUDA graphs. In the code:
  - [`torch_compile_infer.py`](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/train/speed_eval/torch_compile_infer.py): `torch.compile(decoder._step, mode="reduce-overhead")`, then a warm-up over all 7 step lengths so no compile lands inside a timed rollout
  - [`decoding.py`](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/train/speed_eval/decoding.py): the `StaticCache` per layer (fixed shapes, so the step can be a CUDA graph) and `step` / `_step`, where the cache position goes in as a tensor so the graph is not recompiled per position
- **draft_verify**: speculative decoding with the previous action's 7 tokens as the draft. In the code:
  - [`draft_verify_infer.py`](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/train/speed_eval/draft_verify_infer.py): just switches the decoder's `draft` flag on
  - [`decoding.py`](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/train/speed_eval/decoding.py): `tokens` keeps the last action as the draft, `_verify` feeds it in one pass, accepts tokens up to the first disagreement with the argmax, and cuts the cache back
- **threads**: torch's CPU thread pool pinned to 4. The vast ai host showed 24 cores but they were all stale cores.  Capping it to 4 fixed this bug. the image preprocessing took 113 ms with the default threads and 0.5 ms with one.

### Harness level optimizations 

- **pipeline**: three copies of the inference loop run in parallel threads, so a new prediction starts every 100 ms instead of waiting for the previous one to finish. The arm gets an action every control period, but each action was computed from a frame that is one full latency old.
- **smoothing**: an exponential moving average over the applied translation and rotation (alpha 0.5), gripper untouched. Basically a low level post-controller 

Each step timed below:

| stage | vanilla | b16 + compile + draft-verify | same + threads |
|---|---|---|---|
| compose the two cameras | 2 ms | 2 ms | 2 ms |
| preprocess (crop, resize, normalize, tokenize) | ~80 ms | 93 ms | 4 ms |
| model (prefill + action tokens) | ~190 ms | 80 ms | 75 ms |
| end to end, p50 / p99 | 273 / 407 ms | 175 / 291 ms | 82 / 128 ms |


## Results

### First pass: the twelve stacks of the config

This first pass tested a few options with only 9 rollouts and few task combinations. 

9 rollouts per test (red cube on red cylinder, red cylinder on blue cube, blue cube on red cube, 3 layouts each). Lockstep is meant to be the optimal, so it is the ceiling performance for each optimization option.

But also, the real-time metric is the one to look at. In real time performance, the vanilla VLA gets 0/9 success rate because inference takes so long. So if any of these optimizations perform better than 0/9, it is considered a success. 

The combinations are shown below. Some are single optimizations (b4, torch_compile..) and some are combined (b16 + torch_compile, ...)

| stack | lockstep | real time | p50 ms | p99 ms | actions/s | stall | tokens = vanilla (how close it is to vanilla preds) |
|---|---|---|---|---|---|---|---|
| vanilla (bf16) | 4/9 | 0/9 | 271 | 413 | 2.6 | 71% | 100% |
| b4 | 4/9 | 0/9 | 385 | 511 | 2.0 | 77% | 60% |
| b16 | 4/9 | 0/9 | 203 | 324 | 3.2 | 65% | 95% |
| torch_compile | 4/9 | 2/9 | 206 | 330 | 3.1 | 66% | 80% |
| smoothing | 4/9 | 0/9 | 257 | 391 | 2.7 | 71% | 100% |
| draft_verify | 4/9 | 1/9 | 211 | 353 | 3.1 | 66% | 100% |
| pipeline (3 engines) | 4/9 | 0/9 | 454 | 703 | 6.3 | 39% | 100% |
| b16 + torch_compile | 3/9 | 2/9 | 177 | 300 | 3.5 | 62% | 90% |
| b16 + draft_verify | 4/9 | 2/9 | 159 | 294 | 3.7 | 60% | 90% |
| b16 + torch_compile + draft_verify | 5/9 | 2/9 | 142 | 273 | 3.9 | 57% | 90% |
| b16 + torch_compile + draft_verify + smoothing | 4/9 | 3/9 | 137 | 276 | 4.0 | 56% | 90% |
| b16 + draft_verify + pipeline | 4/9 | 2/9 | 117 | 292 | 9.9 | 6% | 90% |

### Latency Analysis

![success against the latency on the clock](images/task3/latency_sweep.png)

| fixed latency | 0 ms | 50 ms | 100 ms | 200 ms | 300 ms |
|---|---|---|---|---|---|
| vanilla success | 4/9 | 4/9 | 3/9 | 1/9 | 1/9 |
| stall | 0% | 29% | 45% | 63% | 72% |

As mentioned earlier, 0 ms (Instant inference) reproduces lockstep exactly. The policy tolerates about 50 ms of waiting per action before degrading from 100 ms on. So in theory, if any optimization can make inference <50ms while matching vanilla preds, it will improve performance. This is measured on the 9 rollouts.

### Second pass: 30 rollouts, all six pairs
This second pass does 30 rollouts per option, on all 6 task combinations. More in depth.


| stack | lockstep | real time | p50 ms | p99 ms | actions/s | stall | staleness ms | tokens = vanilla |
|---|---|---|---|---|---|---|---|---|
| vanilla (bf16) | **22/30 (73%)** | 0/30 | 273 | 407 | 2.6 | 72% | 278 | 100% |
| b16 + torch_compile + draft_verify | 17/30 | **14/30** | 175 | 291 | 3.6 | 60% | 176 | 90% |
| same + smoothing | 16/30 | 6/30 | 178 | 288 | 3.6 | 61% | 191 | 90% |
| b16 + draft_verify + pipeline | 19/30 | 11/30 | 116 | 248 | 9.9 | 6% | 115 | 90% |
| vanilla + threads | 22/30 | 10/30 | 195 | 240 | 3.3 | 63% | 195 | 100% |
| **b16 + torch_compile + draft_verify + threads** | 17/30 | **18/30 (60%)** | **82** | **128** | 5.3 | 41% | 82 | 90% |
| b16 + draft_verify + pipeline + threads | 19/30 | 7/30 | 116 | 229 | 10.0 | 8% | 116 | 90% |

(the lockstep column of the three "+ threads" rows is the lockstep result of the same stack without threads: threads changes no output, so lockstep was not rerun)

The lockstep vanilla row is the Task 2 number again (73%). Deployed as is with no lockstep, the same policy scores 0 of 30. Pinning the CPU threads alone, which changes nothing in the model, takes vanilla to 195 ms and 10 of 30 (the preprocessing overhead explained above). The compiled, speculative pred bf16 stack with the thread fix runs at 82 ms p50 and 128 ms p99, 3.3x faster than vanilla end to end, and obtains 18/30 in real time against its own lockstep ceiling of 17/30 (22/30 for vanilla), so for this stack real time is the ceiling. Smoothing the actions costs success (6 of 30) instead of helping, and the three-engine pipeline reaches the full 10 Hz with under 8% stall but caps out at 7 to 11 of 30. Every action is based on a frame one step old so this must harm performance.

## Videos
The red flashing bar signifies a stall, waiting for inference.


Vanilla, real time: the arm waits 273 ms after every 100 ms of motion and never settles the cube
![vanilla real time](images/task3/realtime_vanilla__red_cube_on_red_cylinder__layout5.gif)

bf16 + torch.compile + draft-verify, real time: success in 14 s
![compile and draft-verify real time](images/task3/realtime_compile_draft__red_cube_on_red_cylinder__layout5.gif)

the same with the CPU thread fix, 82 ms per action, real time: success in 11 s
![compile, draft-verify and threads, real time](images/task3/realtime_compile_draft_threads__red_cube_on_red_cylinder__layout5.gif)

draft-verify + 3-engine pipeline, real time: success in 9 s, no visible stall
![pipeline real time](images/task3/realtime_pipeline__red_cube_on_red_cylinder__layout5.gif)

Vanilla, lockstep (the ceiling, no red flashes by design): success in 7 s
![vanilla lockstep](images/task3/lockstep_vanilla__red_cube_on_red_cylinder__layout5.gif)

