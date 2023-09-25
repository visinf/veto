# DATASET

## Visual Genome
The following is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs).

Note that our codebase intends to support attribute-head too, so our ```VG-SGG.h5``` and ```VG-SGG-dicts.json``` are different with their original versions in [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). We add attribute information and rename them to be ```VG-SGG-with-attri.h5``` and ```VG-SGG-dicts-with-attri.json```. The code we use to generate them is located at ```datasets/vg/generate_attribute_labels.py```. Although, we encourage later researchers to explore the value of attribute features, in our paper "Unbiased Scene Graph Generation from Biased Training", we follow the conventional setting to turn off the attribute head in both detector pretraining part and relationship prediction part for fair comparison, so does the default setting of this codebase.

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `pysgg/config/paths_catelog.py`. 
2. Download the [scene graphs](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfI9vkdunDpCqp8ooxoHhloBE6KDuztZDWQM_Sbsw_1x5A?e=fjTSvw) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `pysgg/config/paths_catelog.py`.
3. Link the image into the project folder
```
ln -s /path-to-vg/VG_100K datasets/vg/stanford_spilt/VG_100k_images
ln -s /path-to-vg/VG-SGG-with-attri.h5 datasets/vg/stanford_spilt/VG-SGG-with-attri.h5
```
## GQA Dataset:
The following is adapted from [SHA + GCL](https://github.com/dongxingning/SHA-GCL-for-SGG).
1. Download the GQA images [Full (20.3 Gb)](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). Extract these images to the file `datasets/gqa/images`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Annotations for GQA200 split can be downloaded from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kwwKFbdBB3ZU3c49?e=06qeZc), and put all three files to  `datasets/gqa/`.

## Depth Map Generation for VG and GQA Datasets:
1. Depth maps for both the datatsets are generated using the monocular depth estimator [AdelaiDepth](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS).
2. Alternatively, download [VG-Depth.v2](https://drive.google.com/file/d/13QKtd-ZjrG0K8mBsWk41notleQWSeeYz/view?usp=sharing)
