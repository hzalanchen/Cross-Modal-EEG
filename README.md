# Visual Neural Decoding via Improved Visual-EEG Semantic Consistency

**[Overview]** We construct a **Joint Semantic Space** to align visual image and EEG representations, aiming to alleviate the modality gap introduced by directly projecting EEG features into the CLIP image embedding space. To further enhance cross-modal semantic consistency, we propose two additional strategies: (1) a **Visual-EEG Semantic Decouple Network (VE-SDN)**, an explicit semantic decoupling framework equipped with mutual information-based disentanglement and cycle-consistent reconstruction, and (2) an **Intra-Class Consistency Constraint**, a class-level alignment mechanism that encourages image embeddings within the same category to move closer to their corresponding EEG prototype.



# Environment
- Python=3.10
- torch=2.0.0


# Run Code

```bash
# for only contrastive learning
python train_contrastive.py
# contrastive learning + geometric loss
python train_contrastive.py --geo-loss True --lambda1 0.5

# similar commands for semantic decouple contrastive learning
python train_decouple.py
# decouple + geometric loss
python train_decouple.py --geo-loss True --lambda1 0.5
```


## Citation
If you find the code useful please consider citing our paper:
```
@article{chen2024visual,
  title={Visual Neural Decoding via Improved Visual-EEG Semantic Consistency},
  author={Chen, Hongzhou and He, Lianghua and Liu, Yihang and Yang, Longzhen},
  journal={arXiv preprint arXiv:2408.06788},
  year={2024}
}
```



