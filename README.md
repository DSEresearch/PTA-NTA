# Digital Pathology AI for Morphological Biomarker Discovery: Tracing CLDN18.2 Expression to Optimize VYLOY Treatment Strategies in Gastric Cancer
## Introduction

Gastric cancer remains one of the leading causes of cancer-related mortality worldwide, despite recent advances in targeted therapeutics and precision oncology. Among emerging targeted treatments, VYLOY has demonstrated promising clinical outcomes for patients with CLDN18.2-positive gastric tumors. However, current clinical workflows rely heavily on immunohistochemistry (IHC) staining platforms such as Ventana Medical Systems to identify CLDN18.2 expression, which introduces additional cost, processing time, and dependency on specialized pathology infrastructure. Furthermore, existing digital pathology studies primarily focus on statistical correlations between hematoxylin and eosin (H&E) slides and clinical outcomes, often without explicitly modeling the underlying biological mechanisms associated with biomarker expression.

This research proposes a next-generation Digital Pathology AI framework that directly learns the morphological fingerprints associated with CLDN18.2 expression rather than relying solely on outcome-based correlations. The central hypothesis of this work is that CLDN18.2 expression leaves latent spatial and cellular morphological signatures within tissue architecture that can be identified and traced through artificial intelligence. By learning these morphology-driven biomarker patterns, the proposed framework aims to infer CLDN18.2 expression without requiring additional Ventana staining procedures, enabling stain-free biomarker prediction using standard H&E whole-slide images.

Unlike conventional patch-level classification approaches, this study focuses on spatial biomarker tracing through cell-centered graph representations and topology-aware modeling. The proposed framework analyzes not only the presence of CLDN18.2-positive regions but also their density, spatial distribution, clustering behavior, glandular organization, and microenvironmental relationships across entire whole-slide images. Such spatial heterogeneity may provide clinically meaningful information regarding treatment response and therapeutic optimization for VYLOY administration. Ultimately, this work seeks to establish a biologically informed precision oncology framework that bridges digital pathology, morphology-aware artificial intelligence, and targeted treatment strategy modeling.

## Methodologies

The proposed framework begins with whole-slide images acquired from Ventana iSyntax digital pathology systems containing paired CLDN18.2-positive and negative gastric cancer slides. The iSyntax files are converted into multi-resolution Zarr representations to enable scalable high-resolution analysis. Tissue segmentation is then performed to remove background regions and artifacts, followed by quality control procedures that eliminate blurry or low-information regions. To capture cellular-level morphology, HoverNet is employed for nuclei segmentation and localization, enabling the extraction of cell-centered image patches at multiple spatial scales.

To prevent the artificial intelligence model from learning shortcut features associated with brown chromogenic staining rather than true morphology, the framework incorporates morphology-preserving stain removal and virtual H&E transformation procedures. The transformed patches are subsequently processed using pathology foundation models, including CONCH and Virchow2, to generate high-dimensional morphology embeddings that encode cellular structure, tissue organization, and microenvironmental context. These embeddings serve as node features for graph construction, where spatial relationships between neighboring cells are modeled through graph-based representations.

The constructed cell graphs are analyzed using Graph Neural Networks (GNNs) and Graph Transformers to learn CLDN18.2-associated spatial biomarker topology. Rather than predicting binary labels alone, the proposed models aim to trace biomarker density, spatial heterogeneity, glandular aggregation, tumor-edge localization, and cellular neighborhood interactions throughout the tissue. The learned representations are further utilized to generate virtual biomarker heatmaps that visualize predicted CLDN18.2 expression patterns across whole-slide images. Finally, the extracted spatial biomarker characteristics are correlated with VYLOY treatment strategies to investigate how morphology-driven biomarker organization may support therapeutic decision-making, treatment timing optimization, and precision oncology workflows.

## Output
<img width="1102" height="638" alt="cldn_output" src="https://github.com/user-attachments/assets/eafd462d-28f8-4de0-b71d-8339be40dece" />

<!--
# Pipeline
1. convert_isyntax_to_zarr.py
   
   * Input:  ./slides/*.isyntax
   * Output: ./zarr/[Slide_ID].zarr
           zarr_conversion_logs.csv

3. zarr_clean_manifest.py
   
   * Input:  ./zarr/[Slide_ID].zarr
   * Output: clean_manifest.csv
           QC images for checking main tissue/noise detection

5. dinov2_final.py
   
   * Input:  clean_manifest.csv
   * Output: DINOv2 training/features using only clean 518×518 Level 0 patches

# Installation on Ubuntu
* Ubuntu 20 on Kakao Cloud with 8 B200 GPUs.
* 200 iSyntax files with VYLOY-Based Features

* sudo sed -i 's|http://ftp.daum.net/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
* sudo add-apt-repository ppa:deadsnakes/ppa
* sudo apt update
* sudo apt install python3.8 python3.8-venv python3.8-distutils
* sudo apt install libgles2 libegl1
* sudo apt install liblcms2-2
* sudo apt install libtinyxml2.6.2v5 libgles2-mesa libegl1-mesa
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-python3-softwarerendercontext_5.1.0-1_all.deb
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-pixelengine_5.1.0-1_amd64.deb 
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-python3-pixelengine_5.1.0-1_all.deb
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-softwarerenderer_5.1.0-1_amd64.deb 
* #PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages
* python3.8 -c "import pixelengine; print('pixelengine OK')"
* python3.8 -c "import pixelengine; print(pixelengine.__file__)"
* #export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
* #export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
* python3.8 -m pip install opencv-python-headless
* sudo apt-get install -y libgl1-mesa-glx
* sudo apt --fix-broken install
 
# Execution Steps
- pyton3.8 convert_isyntax_to_zarr.py

- pyton3.8 zarr_clean_manifest.py

- python3.11 -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  dinov2_final.py \
  --manifest ./clean_manifest.csv \
  --output-dir ./dinov2_features \
  --mode extract \
  --backbone dinov2_vitb14 \
  --dinov2-repo /home/jovyan/dinov2 \
  --tile-size 518 \
  --batch-size 32 \
  --num-workers 0
-->


*Sponsored by GensionBio*
