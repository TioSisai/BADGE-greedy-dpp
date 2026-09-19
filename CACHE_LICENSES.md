# Cache licenses and sources

This notice covers the `HyenaSET/` frame-level cache for this repository. The [accompanying paper](https://arxiv.org/abs/2607.13555) uses about 205 hours of fully annotated audio with 10 sound classes from a preliminary version of HyenaSET.

[HyenaSET](https://doi.org/10.64898/2026.06.14.732108) by Woerner et al. contains collar recordings from 19 spotted hyenas (*Crocuta crocuta*) in the Masai Mara National Reserve, Kenya, collected by the Mara Hyena Project of Michigan State University and the Max Planck Institute of Animal Behavior.

The [public dataset](https://doi.org/10.17617/3.8ZSP3J), V1.0 published on Edmond on 22 June 2026, is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/), as specified in its [official terms of use](https://edmond.mpg.de/api/datasets/:persistentId/versions/1.0/customlicense?persistentId=doi:10.17617/3.8ZSP3J).

This license permits noncommercial sharing and adaptation. Redistribution must retain creator attribution, source and license links, and supplied rights notices, identify modifications, and share adapted material under the same or a compatible license.

The public license record covers V1.0, which includes about 243 hours of reviewed annotations. It does not independently establish redistribution authorization for the earlier, approximately 205-hour version used by this cache.

For data attribution, cite Woerner et al. (2026), *HyenaSET: Hyena Sound Event Transcripts*, Edmond, V1.0, [doi:10.17617/3.8ZSP3J](https://doi.org/10.17617/3.8ZSP3J), and the [HyenaSET paper](https://doi.org/10.64898/2026.06.14.732108).

Feature extraction used [animal2vec](https://doi.org/10.1111/2041-210X.70218) by Schaefer-Zimmermann et al., *Methods in Ecology and Evolution* 2026, 17(3):875-888.

Shiqi Zhang, Marius Faiß, Ariana Strandburg-Peshkin, and Tuomas Virtanen prepared this cache for the accompanying paper. Processing includes 10-second segmentation, animal2vec embedding extraction, mean pooling of every 100 frames into 20 frames of 0.5 s, frame-level and segment-level label conversion, and a 70/15/15 split stratified by segment-level label.

Raw audio and pretrained weights are not included. Repository code is released under the [MIT license](LICENSE).
