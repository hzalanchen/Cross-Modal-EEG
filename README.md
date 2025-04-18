# Visual Neural Decoding via Improved Visual-EEG Semantic Consistency

**[Abstract]** Visual neural decoding refers to the process of extracting and interpreting original visual experiences from human brain activities. Recent advances in metric learning based electroencephalogram (EEG) visual decoding methods have delivered promising results and demonstrated the feasibility of decoding novel visual categories from brain activities. However, the methods that directly map EEG features to a pre-trained model embedding space may introduce mapping bias and cause semantic inconsistency among features, thereby impairing modality alignment. To explore the semantic consistency between visual and neural signals, this work constructs a joint semantic space and proposes a Visual-EEG Semantic Decouple Framework to facilitate the optimal modality semantic alignment. Specifically, a cross-modal information decoupling module is introduced to guide the extraction of semantic-related information from modalities. By quantifying the mutual information between visual image and EEG features, we observe a strong positive correlation between decoding performance and the magnitude of mutual information. Inspired by the mechanisms of visual object understanding from neuroscience, this work proposes an intra class geometric consistency approach for the alignment process. This strategy maps visual samples within the same class to consistent neural patterns, further enhancing the robustness and performance of EEG visual decoding. Experiments on a large Image-EEG dataset show that our method achieves state-of-the-art results in zero-shot neural decoding tasks.
<p align="center">
<img src="fig/VE_SID.png" width="600" class="center">
</p>


# Dependencies
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



