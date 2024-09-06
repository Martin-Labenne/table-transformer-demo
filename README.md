# table-transformer-demo

To configure this project you will need to have conda installed, if its not already the case, you can visit the miniconda [installation page](https://docs.anaconda.com/miniconda/)

This project is meant to be a simple picture to spredsheet microservice wrapping the project [table transformer](https://github.com/microsoft/table-transformer) from microsoft. It use both the original project repository and the models available on [HuggingFace](https://huggingface.co/collections/microsoft/table-transformer-6564528e330b667bb267502e). 

This is still work in progress ^^ 

## Init the project

```bash
python setup.py
```

This script will 
- clone table transformer's project from microsoft and fix it to a given commit (at the time of writing this is main) `16d124f616109746b7785f03085100f1f6247575`
- create a conda environment name `table-transformer` with the needed dependencies

After executing this script, you will be able to activate the environment with
```bash
conda activate table-transformer
```

Finally, you will need to copy create a `.env` file from the template `.env.default` 
