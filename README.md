# PIXELS - Progressive Image Xemplar-based Editing with Latent Surgery
>  Shristi Das Biswas, Matthew Shreve, Xuelu Li, Prateek Singhal, Kaushik Roy 
> <br> Purdue University, Amazon.com  
> Recent advancements in language-guided diffusion models for image editing are often bottle-necked by cumbersome prompt engineering to precisely articulate desired changes. A more intuitive alternative calls on guidance from in-the-wild exemplars to help users draw inspiration and bring their imagined edits to life. Contemporary exemplar-based editing methods shy away from leveraging the rich latent space learnt by pre-existing large text-to-image (TTI) models and fall back on training with curated objective functions to achieve the task, which though somewhat effective, demands significant computational resources and lacks compatibility with diverse base models and arbitrary exemplar count. On further investigation, we also find that these techniques enable user control over the degree of change in an image edit limited only to global changes over the entire edited region. In this paper, we introduce a novel framework for progressive exemplar-driven editing with off-the-shelf diffusion models, dubbed PIXELS, to enable customization by providing granular control over edits, allowing adjustments at the pixel or region level. Our method operates solely during inference to facilitate imitative editing, enabling users to draw inspiration from a dynamic number of reference images and progressively incorporate all the desired edits without retraining or fine-tuning existing generation models. This capability of fine-grained control opens up a range of new possibilities, including selective modification of individual objects and specifying gradual spatial changes. We demonstrate that PIXELS delivers high-quality edits efficiently, outperforming existing methods in both exemplar-fidelity and visual realism through quantitative comparisons and a user study. By making high-quality image editing more accessible, PIXELS has the potential to enable professional-grade edits to a wider audience with the ease of using any open-source generation model.


<a href="to-be-filled"><img src="https://img.shields.io/badge/arXiv-2306.00950-b31b1b?style=flat&logo=arxiv&logoColor=red"/></a>
<a href="to-be-filled"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a>
<br/>
<img src="assets/teaser.png" width="800px"/>  

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)


## Requirements

- Python (version 3.9)
- GPU (NVIDIA CUDA compatible)
- [Virtualenv](https://virtualenv.pypa.io/) (optional but recommended)

## Installation

- Create a virtual environment (optional but recommended):

    ```conda create --name pixels```

- Activate the virtual environment:

    ```conda activate pixels```

- Install the required dependencies:

    ```pip install -r requirements.txt```

## Usage
- Ensure that your virtual environment is activated.
- Make sure that your GPU is properly set up and accessible.

- For Stable Diffusion XL:
  - Run the script:

    ```python SDXL/inference.py```


- For Stable Diffusion 2.1:
  - Run the script:

    ```python SD2/inference.py```


- For Kandinsky 2.2:
  - Run the script:

    ```python Kandinsky/inference.py```

    
## Citation
```
to be filled
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
