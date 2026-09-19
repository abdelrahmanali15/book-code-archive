# Setup Instructions

Here you will find several methods of running the code found in the book. There are two preferred methods, one for local usage and one for cloud-based:

* **Local**: Using a [Conda](../.setup/conda) environment
* **Cloud**: Using [Google Colab Notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main?tab=readme-ov-file#table-of-contents)

## Quick start

If you already have a local Python (3.10.*) environment and [Microsoft Visual C++ 14.0](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or greater installed, you can install the complete environment as follows at the root of this repository:

```bash
pip install -r requirements.txt
```

> [!WARNING]
> If you get the following error `error: Microsoft Visual C++ 14.0 or greater is required.` then you will need to install C++. 
> Follow the instructions [here](../.setup/conda/common_issues.md) for an installation guide before you can install your environment.

If you have conda installed (this **does not** require an additional installation of C++), you can also install the complete environment as follows at the root of this repository:

```bash
conda env create -f environment.yml
```

> [!TIP]
> After preparing your environment, it is recommended to install the GPU version of PyTorch following the instructions [here](https://pytorch.org/) as most examples will require a GPU.

For a complete tutorial on setting up your environment, visit the [conda example](../.setup/conda).
