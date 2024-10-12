---
layout: post
title: "Twelve Projects"
date: 2021-12-30
published: false
hidden: true
---

## Big Ideas
This is a living list of the big problems I'm working on now. They are written here to keep track, such that as new research comes out we can see how these problems may be solved.

- Asynchronous event processing with FPGA tools. Stereo vision, blob tracking.
- Solving event-based perception with deep learning methods.
- Scaling tarnsformers and deep neural networks with current CMOS hardware.

### Event-Based Processing
- How do you efficiently match points in space with stereo DVS cameras? What is the underlying structure you can take advantage of?
- How do you track fast blobs in real time with a DVS camera? What is the optimal use case for such technology?
- Are there ways deep learning is superior over neuromorphic processing for dealing with DVS data? How can we show that in a paper?
- Are there ways neuro-inspired async processing can outperform deep learning and Loihi?

### Deep Learning Acceleration
After reading gwern's "AI Scaling" article and further investigating GPT-3 and large transformers, I'm becoming convinced that the greatest thing holding AI back is a lack of hyper-scale model training. Current setups used to train the largest models, which all happen to be transformers, use GPUs or TPUs. While GPUs have been outfitted with processing elements that improve performance training models, they still aren't built solely for the purpose of training. TPUs improve on this, though their locked-down nature and difficulty of programming render them suboptimal. I believe that an AI chip company with values and architectures such as Tenstorrent, where the atomic operation is a matrix multiply, which consists of many (as many as you want) processing elements, is the best way to leverage current bit-smashing hardware to accelerate and scale DL. 

For this reason, my current goals in this PhD program are constructed to make me employable at the type of company that will be at the forefront of ML scaling. These include Comma.ai, OpenAI, and Tenstorrent. My place will be that of a deep learning hardware-scalability expert. This job will require one who is familiar with deep learning models and algorithms, is passionate for AI, has experience with PyTorch's backend graph representation, has experience creating accelerators in HLS and Verilog, understands computers from a systems level, and can bring together the model architecture, compiler, matrix multiplication units, networking, and distributed computation of the system. 

Becoming this PhD graduate will require a specific course of action. On the deep learning front, I will have to be familiar with the networks of choice (by reading arxiv), understand training methods (fast.ai), and predict the future path of the field. On the hardware front, I will have to put myself on projects that involve accelerating the training of models. I will not limit myself to transformers, but they are the favorable place to start based on their scalability. Eventually, this could lead to a system-level project that would basically result in a version of the first product Tenstorrent ever demoed, their [one-unit accelerator](https://images.anandtech.com/doci/16709/3%20-%20HotChips2020_FPGA_Tenstorrent-v01-page-027.jpg). 

- How do you create an open-source, scalable DL accelerator like Tenstorrent?


### 
## Unordered and Unpruned
- tracking blobs with event camera and FPGA asynchronously for trajectory estimation
- stereo vision correspondence with event cameras and FPGA parallel processing
- event stream autoencoder with transformer to pretrain encoder
- distributed matrix multiplication using multiple FPGAs, optimizing size of matrix tile operation
- train a transformer on an FPGA or multiple FPGAs with ASIC realization in mind
- compiler whose input is PyTorch graph and output is configuration of distributed matrix multiplication processor
