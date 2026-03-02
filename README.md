# Visual Neural Decoding via Improved Visual-EEG Semantic Consistency

**[Overview]** We construct a **Joint Semantic Space** to align visual image and EEG representations, aiming to alleviate the modality gap introduced by directly projecting EEG features into the CLIP image embedding space. To further enhance cross-modal semantic consistency, we propose two additional strategies: (1) a **Visual-EEG Semantic Decouple Network (VE-SDN)**, an explicit semantic decoupling framework equipped with mutual information-based disentanglement and cycle-consistent reconstruction, and (2) an **Intra-Class Consistency Constraint**, a class-level alignment mechanism that encourages image embeddings within the same category to move closer to their corresponding EEG prototype.



## Environment Setup
You can quickly set up the required dependencies using the provided `env.yml` file.
```bash
# 1. Create the virtual environment
conda env create -f env.yml 
# 2. Activate the environment
conda activate BCI
```

### Key Dependencies
If you prefer to install the environment manually, the core dependencies are listed below for reference:

| Package | Version |
| :--- | :--- |
| **Python** | 3.10.14 |
| **PyTorch** | 2.7.0 |
| **Torchvision**| 0.22.0 |
| **Transformers**| 4.56.0 |

**Other requirements:** `braindecode`, `einops`, `numpy`.





<!-- 
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
-->



