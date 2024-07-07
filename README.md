#table-transformer-demo

## Init the project

This project use table-transformer project as well as table transformers models from hugging face. I suggest to lock your version of TATR project, at the time I write this main is at commit `16d124f616109746b7785f03085100f1f6247575`

```
git clone git@github.com:microsoft/table-transformer.git tableTransformer
cd tableTransformer
git reset --hard 16d124f616109746b7785f03085100f1f6247575


conda env create -f environment.yml
conda activate tables-detr
conda install conda-forge::transformers
conda install -c conda-forge jupyterlab
conda install easyocr
conda install python-dotenv
```
