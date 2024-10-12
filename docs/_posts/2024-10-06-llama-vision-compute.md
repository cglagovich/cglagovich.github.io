---
layout: post
title: "Analyzing Llama3.2-Vision Performance"
date: 2024-10-12
---

This post estimates the perfomance characteristics of the Llama3.2-Visions models. 

To recap, Llama3.2-Vision came in 2 sizes, 11B and 90B. 11B builds on the 8B text model, and 90B builds on the 70B text model. The `CrossAttentionTransformerVision` part of the model is exactly the same for both 11B and 90B, except for the fact that 11B has a vision chunk size of 448 and 90B has a chunk size of 560.

If 11B and 90B have the same vision model, why does 90B have 20B more parameters than 70B, while 11B has 3B more parameters than 8B? That's because in the `CrossAttentionTransformerText` part of the model, 11B and 90B have a different number of cross attention layers -- 8 vs 20. 

# Model Hyperparameters
Let's dig into the performance of 11B and then generalize to 90B.

Important 11B parameters can be found in the [sku list](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py), but it doesn't tell the whole story.
```
"dim": 4096,
"n_layers": 32,
"n_heads": 32,
"n_kv_heads": 8,
"vocab_size": LLAMA3_VOCAB_SIZE,
"ffn_dim_multiplier": 1.3,
"multiple_of": 1024,
"norm_eps": 1e-05,
"rope_theta": 500000.0,
"use_scaled_rope": True,
"vision_chunk_size": 448,
"vision_max_num_chunks": 4,
"vision_num_cross_attention_layers": 8,
```
We need a few more parameters which are hard-coded into the vision encoder:
```
patch_size = 14,
width = 1280,
layers = 32,
heads = 16,
mlp_ratio = 4.0,
in_channels = 3,
n_global_layers = 8,
head_dim = 1280//16 = 80
```

With these parameters, let's figure out which layers the vision model is actually running. Note that I'm going to ignore any embedding or conv layers since their weights and flops are so small.

# The VisionEncoder
The `VisionEncoder` consists of a local transformer and a global transformer. The local transformer has 32 layers and the global has 8 layers. They are constructed with the same dimensions, so let's imagine they are just 40 layers of transformer blocks.

One transformer block contains
```
self.attn = ImageAttention(
    dim=width,
    head_dim=head_dim,
    n_heads=self.n_heads,
)
self.mlp = ImageFeedForward(
    dim=d_model,
    hidden_dim=int(mlp_ratio * d_model),
    dropout=0.0,
    act_layer=act_layer,
)
```

If we expand this, one transformer block contains weights of shape
```
wq: 1280 x 1280
wk: 1280 x 1280
wv: 1280 x 1280
wo: 1280 x 1280
c_fc: 1280 x 5120
c_proj: 5120 x 1280
```

In addition, at the end of the `VisionEncoder` we run an "LM head" on the vision tokens to project the output (plus a few saved intermediate outputs) from 7680 -> `dim` (depending on the dim of the text model). We can ignore this since it's effectively just one more matmul.


# FLOPs and Memory
Now let's compute vision encoder flops and memory BW. I'm going to assume bfloat16 activations and weights. I also want to assume that the image is of full size (4 tiles) and there is only one image in the batch. So the image activations is of shape `[batch=1, num_concurrent_media=1, chunks * ntok = (4 * 1032) = 4128, width=1280]` or more simply `[4128, 1280]`. 

Two helpful equations for us:
- MM FLOPs = `M * K * N * 2`
- MM DataMovement = `(M * K + K * N + M * N) * dtype_bytes`. 
- FlashAttention DataMovement = `(S * D) * 4 * dtype_bytes`, assuming perfect reuse

```
# FLOPS per transformer block
# Projections in attention
wq: 4128 * 1280 * 1280 * 2 = 13,526,630,400
wk: 4128 * 1280 * 1280 * 2 = 13,526,630,400
wv: 4128 * 1280 * 1280 * 2 = 13,526,630,400
# Attention scores
wq @ wk: 4128 * 1280 * 4128 * 2 = 43,623,383,040
# Attention output
scores @ wv: 4128 * 4128 * 1280 * 2 = 43,623,383,040
wo: 4128 * 1280 * 1280 * 2 = 13,526,630,400

# MLP projections
c_fc: 4128 * 1280 * 5120 * 2 = 54,106,521,600
c_proj: 4128 * 5120 * 1280 * 2 = 54,106,521,600
```
This sums to `249,566,330,880` FLOPs per transformer layer.

```
# Data moved per transformer block
# Projections in attention
wq: (4128 * 1280 + 1280 * 1280 + 4128 * 1280) * 2 = 24,412,160
wk: (4128 * 1280 + 1280 * 1280 + 4128 * 1280) * 2 = 24,412,160
wv: (4128 * 1280 + 1280 * 1280 + 4128 * 1280) * 2 = 24,412,160
# FlashAttention
attn_out: (4128 * 1280) * 4 * 2 = 42,270,720

wo: (4128 * 1280 + 1280 * 1280 + 4128 * 1280) * 2 = 24,412,160

# MLP projections
c_fc: (4128 * 1280 + 1280 * 5120 + 4128 * 5120) * 2 = 65,945,600
c_proj: (4128 * 5120 + 5120 * 1280 + 4128 * 1280) * 2 = 65,945,600
```
This sums to `271,810,560` Bytes moved per transformer layer.

If we put it in a table:
| n layers | GFLOPs | MB moved | 
| -------- | ------ | -------- |
| 1 | 249 | 271 |
| 40 | 9,982 | 10,872|

We see that 40 layers of vision transformer blocks will require 9.9 TFLOPs of compute and 10.8 GB of data moved. This is good news for hardware which is close to a 1.0 compute/BW ratio!


# Performance Estimate on an n300
So now we know the compute and memory requirements to run this "vision prefill" of the 11B and 90B models, but it would be helpful to apply these calculations to real hardware to get a feel for how fast the vision encoder can be. 

Let's assume that [we have hardware](https://tenstorrent.com/hardware/wormhole) with 466 TFLOPs and 576 GB/s. Call this hardware n300.

| Compute time (ms) | Memory time (ms) | Total time (ms)
| --- | --- | --- |
| 9.9 TFLOPs / 466 TFLOPs/s | 10.8 GB / 576 GB/s |  |
| 21.4 | 18.9 | 21.4 |

On this n300 hardware, the VisionEncoder will be (just barely) compute bound, running in 21.4 ms! Keep in mind that we're assuming 100% compute and memory utilization, which is impossible -- the real latency will certainly be higher, depending on how efficient your kernels are. Remember that model training runs aim to hit >40% MFU, so that's a good number to use in your calculations. 

We can roughly estimate that Llama3.2-11B's TTFT time will be 21.4 ms longer than Llama3.1-8B. Something we are ignoring is the impact of the 8 cross attention layers in the text model, but that can be a topic for another post. 