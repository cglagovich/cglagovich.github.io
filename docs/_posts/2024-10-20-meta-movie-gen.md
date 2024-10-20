---
layout: post
title: "How fast is Meta Movie Gen?"
date: 2024-10-20
---

On October 4th, Meta [released a paper announcing Movie Gen](https://ai.meta.com/static-resource/movie-gen-research-paper), a new set of media generation foundation models. This follows OpenAI's SORA hype from Februrary 15th of this year. In contrast to the SORA announcement, the Movie Gen paper contains real details on the model architecture, size, and training pipeline. It is unclear whether Meta will release Movie Gen open source, but we can analyze the paper and get some insights into 1) model architecture and 2) AI inference compute demand.

Let's start by contrasting what we know about SORA and Movie Gen.

| Category | SORA | Movie Gen | Notes |
| -------- | ---- | --------- | ----- |
| Modality | Text -> Video | Text -> Video | Ignoring the audio component and the video -> video editing that both models enable. |
| Encoder architecture | Spatio-temporal autoencoder | Spatio-temporal autoencoder | |
| Backbone | Diffusion Transformer | Diffusion Transformer | | 
| Model size | Unknown | 30B ||
| Output size/quality | Variable, HD | Variable, up to 768x768 | Native to the model, not including upsampling |
| Output length | 60s | 16s | |
| Output framerate | Unknown | 16 fps | Native to the model, not including upsampling  |


# Model Details
Meta gave us a lot to work with when they wrote about the model architecture. At the highest level, this is a diffusion transformer, trained on videos rather than images. Diffusion occurs in latent space (this has been state of the art for image generation for a while, since at least [this paper](https://arxiv.org/pdf/2212.09748)). The model consists of a temporal autoencoder (TAE) which maps images/videos into latent space. The diffusion transformer iterates on this latent space, then feeds the output through the TAE decoder to produce images/videos. 

The TAE's input->output shapes are `T' x 3 x H' x W' -> T x C x H x W`. The dimensions T, H, and W are compressed by 8x. In the paper, they mention that they chose C=16. Movie Gen can produce 16 seconds of 16fps, 768x768 px output. That's 256 frames of 768x768 images, which when compressed in the latent space comes to `T=32 x C=16 x H=96 x W = 96`. They note that in real inference, they tile this input in the time dimension with or without overlap to reduce the overhead of the TAE. 

The output of the TAE of shape `T x C x H x W` is patchified via convolution to create tokens for the diffusion model. They use a kernel of size 1 x 2 x 2 with stride of the same dimensions, yielding `T x H/2 x W/2` tokens. For our case of creating 16 seconds, 16fps, 768px video, this becomes `32 x 96/2 x 96/2 = 72k` tokens. 

Now that we have looked at the shape of the latent space we can focus on the diffusion transformer. This is the model which performs `t` timesteps to denoise the latent video. The architecture of the backbone is based on the Llama3 architecture, meaning it uses SwiGLU MLP and RMSNorm. The attention is non-causal (bidirectional) since this is not an autoregressive model. 

The shape of the transformer backbone is described as:
```
n_layers = 48
dmodel = 6144 (6k)
expand_dim = 16384 (16k)
n_heads = 48 (head_dim = 128)
```

Interestingly, this transformer uses multi-head attention (MHA) rather than grouped-query or multi-query (GQA, MQA), which are popular optimizations for LLMs. GQA and MQA are popular because they greatly reduce the size of the KV cache that's read during LLM inference. KV caches can balloon in size for long sequences, so GQA and MQA enable higher throughput and larger batches than MHA would. However, diffusion transformers are not autoregressive so they do not have a KV cache, so they can instead rely on MHA and avoid any hit to capacity/correctness that comes with GQA/MHA.

Each transformer block resembles the Llama3 decoder block, except for the addition of cross attention. The model attends to multiple sources of text conditioning. We will ignore cross attention for the remainder of the post, since we can get interesting insights into the performance of the model without including this variable-shaped cross attention.

Stack these 48 transforer blocks together and you have the full diffusion transformer. Diffusion models iterate for some number of steps to denoise the latent space. Meta found that using 50 steps (with special linear-quadratic schedule) gave good results for video generation.

# Inference Performance Analysis

We already know that diffusion is relatively expensive compared to autoregressive inference (you get ~100 tokens/s for Llama3-70B text generation, but it can take a few seconds to create a single image with diffusion). This is because diffusion pretty much looks like LLM prefill performed in a loop, for however many timesteps the model requires. In addition, diffusion sequence lengths can be relatively long, and the non-causal attention in diffusion transformers adds up to make these models fairly compute bound. 

To analyze inference performance, we first need to figure out the memory bandwidth and flops involved. I'm going to make quite a few assumptions here which will give us an optimistic result. I'll be leaving out 1) the TAE, 2) cross attention on text, and 3) upsampling.

I'll focus on the performance of generating 16 seconds @ 16fps of 768x768px video, which is the most that the model is capable of. We know the shape of the diffusion transformer layers, and that they run 50 timesteps perf video. Let's start with the flops and memory of a single layer.

The input to a layer is of shape `(S=72k, D=6k)`. Let's also say that all activations/weights are in bf16. Recall a few equations we've looked at before:

- MM FLOPs = `M * K * N * 2`
- MM DataMovement = `(M * K + K * N + M * N) * dtype_bytes`. 
- FlashAttention DataMovement = `(S * D) * 4 * dtype_bytes`, assuming perfect reuse

The shapes of each operation in a single block are the following
```
# FLOPS per transformer block
# Projections in attention
wq: (S x D) @ (D x D)
wk: (S x D) @ (D x D)
wv: (S x D) @ (D x D)

# Attention scores
wq @ wk: (NH x S x DH) @ (NH x DH x S)
# Attention output
scores @ wv: (NH x S x S) @ (NH x S x DH)
wo: (S x D) @ (D x D)

# MLP projections
ff1: (S x D) @ (D x ED)
ff3: (S x D) @ (D x ED)
ff2: (S x ED) @ (ED x D)
```

When I throw that into my spreadsheet, I get

| Operation	| GFLOPs | Data (MB) |
| --------- | ------ | --------- |
| wq         | 5566.28  | 1800    |
| wk         | 5566.28  | 1800    |
| wv         | 5566.28  | 1800    |
| scores     | 66795.33 | 3456    |
| attn_out   | 66795.33 | 3456    |
| wo         | 5566.28  | 1800    |
| ff1        | 14843.41 | 3360    |
| ff3        | 14843.41 | 3360    |
| ff2        | 14843.41 | 3360    |

Add those up and we can look at the full model (note that the units changed):

| Operation | TFLOPs | Data (GB) |
| --------- | ------ | --------- |
| Single Layer | 200.3 | 23.6 |
| All (48) Layers | 9618.5 | 1134 |
| All (50) timesteps | 480,926 | 56,700 |

That's pretty huge. To compare, autoregressive transformer inference is usually memory-bound on the order of 10GB data moved per forward pass.

Now consider that I didn't even include the TAE or upsampler, so these numbers are optimistic. However, I am assuming bf16 weights and activations. It is reasonable that an inference provider would quantize to at least fp8, so they could decrease both FLOPs and Data by 2x. 

Still, the perfomance implications of this are crazy. Most LLMs we run are memory bound (in decode mode). Most AI accelerators have a `compute / memory BW` ratio less than 1, meaning they are memory heavy. For Meta Movie Gen inference, the `FLOP / Byte` ratio is `480.9 PFLOP / 56,700 GB  = 7899`! The model performs 7899 floating point operations per byte moved from memory. This is because diffusion transformer inference looks a lot like LLM prefill with a massive sequence length. As the sequence increases, attention FLOPs dominate compute and drive the FLOP/Byte up. 

For the following sections, let's look at how quickly state of the art AI accelerators can run this workload. For each accelerator, I will assume 50% utilization of memory bandwidth and FLOPs. This accounts for issues with suboptimized kernels, latencies due to CCL, and underutilization due to batching or serving. In addition, it's just really tough to get close to 100% utilization on any accelerator.

## 8xH100 SXM

Nominally, an H100 SXM has 1,979 TFLOP/s BF16 and and 3.35 TB/s memory bandwidth. Downrating those by 50% gives us
- 989.5 TFLOP/s BF16
- 1.675 TB/s memory BW

Let's assume we're doing inference on 8xH100 SXM, which is very reasonable for such a large model. That gives us
- 7.9 PFLOP/s BF16
- 13.4 TB/s memory BW

This 8xH100 SXM machine would take `60.8 s` on compute and `4.2 s` on memory, giving us `60.8 s` per video. To look at it another way, this machine could generate **59.2 videos per hour** if there is no downtime between videos.

We see that due to the [H100 bubble bursting](https://www.latent.space/p/gpu-bubble), H100 providers are pricing at ~$2/hour. Let's assume that 8xH100 SXM is $16/hour and there isn't an upcharge to use them together in a DGX. At that rate, the **cost per video is $0.27**. 

Compare that to LLM inference, where Llama3.1-70B is selling for about $0.9/1M tokens [on together.ai](https://artificialanalysis.ai/models/llama-3-instruct-70b/providers). At the speed of together.ai's reference implementation (107 t/s), it would take about 2.6 hours to generate 1M output tokens and spend $0.9. Taking this further, this means that as a single user you can spend $9.63e-5 per second on LLM inference, while you can spend $4.4e-3 per second on video generation. To spend money on LLM inference at the same rate as you could on video generation, you would need to have 51 concurrent LLM users for every 1 video generation user.

## WH Galaxy

Now say that we have a system with the following specs ([all taken from the website's datasheet](https://tenstorrent.com/hardware/galaxy)) (after 50% downrating):
- 4.65 PFLOP/s BFP8 (number should also make sense for BF16)
- 3.84 TB/s memory BW

To run the forward pass of the diffusion transformer, this WH Galaxy would require `103.4 s` on compute and `14.8 s` on memory, giving us `103.4 s` per video. This is very comparable to the 8xH100 SXM estimates we just made. I'll leave it to other people to worry about pricing and perf/$ comparisons. 

# Implications on AI inference market

Now that we've done a small case study of Meta Movie Gen inference on 8xH100 and WH Galaxy, what does it all mean? There are a few takeaways.
- H100 is poorly specced for this workload. It is extremely memory heavy (0.59 PFLOP/TB ratio) because of its expensive HBM
- More compute-focused machines will perform better on video gen. WH Galaxy is a strong contender given its more balanced 1.21 PFLOP/TB compute-to-memory ratio
- Video generation is crazy expensive (roughly 50x more expensive that LLM inference)
- Due to being compute bound, video generation likely does not get cheaper with batching

As these video generation models get better with scaling laws and algorithmic improvements, coordinating the compute of many accelerators will be an increasingly important problem for inference providers. Expensive HBM and networking is going to hurt profits, so commodity AI inference hardware will be necessary to achieve widespread video generation. In addition, overlapping datamovement with compute to achieve high compute utilization will be paramount. Software optimizations such as asynchronous DRAM prefetching and overlapped CCL must be exposed to the programmer or compiled by the framework.

Looking forward, I can imagine a future where high quality video generation models are running in real time. 4k output, 60fps, minutes of video continuously generating simulated worlds for VR or gaming or Netflix. The number of tokens in the diffusion transformer could grow from 72k to 14.5 million tokens, leading to 4-5 OOM increased compute requirements! To generate these high quality videos in real time, the compute of GPUs will have to increase by at least 5 OOM. Bottom line, the world does not have enough compute!
