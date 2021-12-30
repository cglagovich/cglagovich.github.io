layout: page
title: "Twelve Projects"
permalink: /posts/twelve/



#Twelve Projects
## Big Ideas
- How to track blobs in space and on ground with low resources
- How to instantly identify stereo matches in two event streams
- How can transformers be applied to DVS data in a novel, useful way
- How can transformers (and any other general ANN) training scale with DL-specific hardware
- How do you create a learning system that effectively uses current technology - cmos, non-neuromorphic hardware
- How can you use the structure of the brain to inspire ANN architecture? i.e. basal ganglia is a learned CPU that controls the cortex, can you have a learned agent control the flow of information through the rest of the network?
- A language model should understand language, not facts. How do you teach it to query facts and construct a meaningful response from a text source?


##Unordered and Unpruned
- tracking blobs with event camera and FPGA asynchronously for trajectory estimation
- stereo vision correspondence with event cameras and FPGA parallel processing
- event stream autoencoder with transformer to pretrain encoder
- distributed matrix multiplication using multiple FPGAs, optimizing size of matrix tile operation
- train a transformer on an FPGA or multiple FPGAs with ASIC realization in mind
- compiler whose input is PyTorch graph and output is configuration of distributed matrix multiplication processor
