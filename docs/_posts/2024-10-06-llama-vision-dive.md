---
layout: post
title: "Deep dive into Llama3.2 vision models"
date: 2024-10-06
published: false
---

Goals for this post include: 
- Identify the flow of tensors through the forward pass of the model
- Identify the modules used for multimodal Llama

In this post I'll dig into the details of the [reference implementation](https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/multimodal/model.py) that Meta provided for the Llama3.2 multimodal models.


## Flow of inputs through the model
`generation.py` contains model building and inference code which is common for all Llama models. In its `generate` function, we see the first invocation of a vision module:
```python
def generate(...):
    is_vision = not isinstance(self.model, Transformer)
    if is_vision:
        images = model_input.vision.images if model_input.vision is not None else []
        mask = model_input.vision.mask if model_input.vision is not None else []

        # the method works for bsz > 1 so add a batch dimension
        xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = (
            self.model.compute_vision_tokens_masks(
                batch_images=[images],
                batch_masks=[mask],
                total_len=total_len,
            )
        )
```

If we step into [model.py](https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/multimodal/model.py#L1353), we see that the vision part of the model operates before we do any text processing.

```python
def compute_vision_tokens_masks(
    self,
    batch_images: List[List[PIL_Image.Image]],
    batch_masks: List[List[List[int]]],
    total_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    skip_vision_encoder = False

    assert len(batch_images) == len(
        batch_masks
    ), "Images and masks must have the same length"

    max_num_images = max(len(x) for x in batch_images)
    bsz = len(batch_images)

    if max_num_images == 0:
        num_chunks = [[self.max_num_chunks] for _ in batch_images]
        skip_vision_encoder = True
    else:
        images_and_aspect_ratios = [
            [self.image_transform(im) for im in row] for row in batch_images
        ]
        transformed_images = [
            [x[0] for x in row] for row in images_and_aspect_ratios
        ]
        
        aspect_ratios = torch.ones(bsz, max_num_images, 2, dtype=torch.int64)
        for i, row in enumerate(images_and_aspect_ratios):
            if len(row) > 0:
                aspect_ratios[i, : len(row)] = torch.stack(
                    [torch.tensor(x[1]) for x in row]
                )

        stacked_images, num_chunks = _stack_images(
            transformed_images,
            max_num_chunks=self.max_num_chunks,
            image_res=self.params.vision_chunk_size,
            max_num_images=max_num_images,
        )
```
By this point, `transformed_images` contains chunked images for each image in the batch. The model does this chunking according to `vision_chunk_size`, which for the 11B model is 448, meaning that a chunk can be at most 448x448 px. 

```python

```
