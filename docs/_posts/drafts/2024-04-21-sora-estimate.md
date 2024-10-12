---
layout: post
title: "Deconstructing SORA"
date: 2024-04-21
published: false
---

This post goes into my best guess at OpenAI's SORA architecture. We will "delve" into which methods they picked up from the research, inference FLOPs, and future architectural optimizations. 

## Blog post clues
- What do we know for certain
- What can we infer

From the [OpenAI SORA "techincal report"](https://openai.com/research/video-generation-models-as-world-simulators), we get a few hints into the architecture of SORA. We know for certain that SORA feeds 3D (spacetime) patches through a latent diffusion model (LDM) for some number of diffusion steps before decoding these patches into video frames. They use some special positional encoding which allows them to generate videos of arbitrary aspect ratios. That is pretty much all that is certain, so we will have to guess at the rest.

We can take a look at SOTA diffusion models to get hints into SORA. Stable Diffusion 3 goes up to 8B parameters, and it is pretty much image only, so let's take that as a lower limit to SORA size. 

## Proposed architecture
- How many params
- Single shot for all frames?
- How long is context in attention

## FLOPs estimate
- Separate estimate for MLP and attn?

## Future optimizations
- Sparse attention, interleaved temporal and spatial layers
