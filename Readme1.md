# Forget Complex Traditional Approaches to handle NLP Datasets, HuggingFace Dataset Library is your saviour! Part-2


## Authors
**Nabarun Barua**     

[Git](https://github.com/nabarunbaruaAIML)/ [LinkedIn](https://www.linkedin.com/in/nabarun-barua-aiml-engineer/)/ [Towardsdatascience](https://medium.com/@nabarun.barua)

**Arjun Kumbakkara** 

[Git](https://github.com/arjunKumbakkara)/ [LinkedIn](https://www.linkedin.com/in/arjunkumbakkara/)/ [Towardsdatascience](https://medium.com/@arjunkumbakkara)


This is the continuation of ealier Part, here we will discuss some more advance thing if you want to read our [Earlier Publised document.](https://medium.com/mlearning-ai/forget-complex-traditional-approaches-to-handle-nlp-datasets-huggingface-dataset-library-is-your-1f975ce5689f)

In this document we will focus on:
- Merging Dataset
- Caching Dataset
- And Cloud Storage
- How to create Document Retreval System

## Merging Dataset

There may be situations where a Data Scientist may have to Merge multiple Datasets together to form a single dataset. There are two ways by which we can merge datasets:

- **Concatenate:** In this we can merge different Dataset who have same number of columns and share the same column types if axis is zero.

    If axis is one then we can concatenate the two dataset if the number of rows are same in the Dataset.

    Let see with an example (Example taken from huggingface.co)
    ```python
        from datasets import concatenate_datasets, load_dataset

        bookcorpus = load_dataset("bookcorpus", split="train")
        wiki = load_dataset("wikipedia", "20200501.en", split="train")
        wiki = wiki.remove_columns("title")  # only keep the text

        assert bookcorpus.features.type == wiki.features.type
        bert_dataset = concatenate_datasets([bookcorpus, wiki])

    ```

- **Interleave Dataset:** Here we can merge several dataset togather by taking alternate examples from each one to create new dataset. This is called Interleaving.

    This can be used in both in regular Dataset as well as streaming datasets

    In the below Example we are doing for Streaming Dataset(same can be done for Regular Dataset) and giving probabilities in the example which is optional. If probabilities are given then final Dataset is formed based on the same probability.(Example taken from huggingface.co)
    ```python
    from datasets import interleave_datasets
    from itertools import islice
    en_dataset = load_dataset('oscar', "unshuffled_deduplicated_en", split='train', streaming=True)
    fr_dataset = load_dataset('oscar', "unshuffled_deduplicated_fr", split='train', streaming=True)

    multilingual_dataset_with_oversampling = interleave_datasets([en_dataset, fr_dataset], probabilities=[0.8, 0.2], seed=42)
    ```

    Around 80% of the final dataset is made of the en_dataset, and 20% of the fr_dataset.

## Caching Dataset

When you download a dataset, the processing scripts and data are stored locally on your computer. The cache allows 🤗 Datasets to avoid re-downloading or processing the entire dataset every time you use it.

**Caching Directory:** We can change the default cache directory from current directory i.e.  ```~/.cache/huggingface/datasets```. By simply setting the enviorment variable.

```bash
$ export HF_DATASETS_CACHE="/path/to/another/directory"
```

Similarly we can do the same by passing parameter ***cache_dir*** in different keywords like load_dataset,load_metric & etc.

Example

```python
from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20200501.en", split="train",cache_dir="/path/to/another/directory")
```

**Download Mode:** Once Dataset is downloaded it's been cahced therefore again when we load the Dataset it doesn't download from source rather it loads from cache.

Now in case there is any change in Dataset and if we want to load Dataset unchanged dataset from the source then we need to use parameter ```download_mode```.

Example

```python
from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20200501.en", split="train",download_mode='force_redownload')
```

**Clean the Cached Files:** If there is a need to clean-up the cached files then we can do it simply by executing following command
```python
# Below fuction just clears the cache of Dataset
ds.cleanup_cache_files()
```
**Enable or disable caching:** There may be a situation when we don't want to cache. In that situation we can do the disable caching locally or globally.

- Locally: If we’re using a cached file locally, it will automatically reload the dataset with any previous transforms you applied to the dataset. We can disable caching using Parameter ```load_from_cache=False```, in ```datasets.Dataset.map()```.

    Example:
    ```python
    updated_dataset = small_dataset.map(tokenizer_function, load_from_cache=False)
    ```
    In the example above, 🤗 Datasets will execute the function ```tokenizer_function``` over the entire dataset again instead of loading the dataset from its previous state.
- Globally: If we want to disable the Caching Globally then below parameter need to be set for Disable caching on a global scale with ```datasets.set_caching_enabled()```:

    ```python
    from datasets import set_caching_enabled
    set_caching_enabled(False)
    ```
    When you disable caching, 🤗 Datasets will no longer reload cached files when applying transforms to datasets. Any transform you apply on your dataset will be need to be reapplied.

## Cloud Storages:

Huggingface Dataset can be stored to popular Cloud Storage. Hugginface Dataset has in-built feature to cater this need.

List of cloud which it supports and filesystem need to be installed to be directly used in Loading Dataset(Table taken from Huggingface.co):

|Storage provider 			| Filesystem implementation |
|---------------------------|:-------------------------:|
| Amazon S3         	 	| [s3fs](https://s3fs.readthedocs.io/en/latest/)|
| Google Cloud Storage		| [gcsfs](https://gcsfs.readthedocs.io/en/latest/)| 
| Azure DataLake			    | [adl](https://github.com/fsspec/adlfs)|
| Azure Blob				    | [abfs](https://github.com/fsspec/adlfs)|
| Dropbox				        | [dropboxdrivefs](https://github.com/fsspec/dropboxdrivefs)|
| Google Drive			    | [gdrivefs](https://github.com/fsspec/gdrivefs)|

Here we will try to show how to load and save Dataset with s3fs to a S3 bucket. For other clouds please see the documentation. Though other cloud filesystem implementations can be used similarly.

First to install the S3 Dependency with Dataset.
```bash
pip install datasets[s3]
```
**Load Dataset:** Now to access the dataset from private S3 bucket by entering your aws_access_key_id and aws_secret_access_key

```python
import datasets
s3 = datasets.filesystems.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# load encoded_dataset to from s3 bucket
dataset = load_from_disk('s3://a-public-datasets/imdb/train',fs=s3)
```
**Save Dataset:** After you have processed your dataset, you can save it to S3 with ```datasets.Dataset.save_to_disk()```:

```python
import datasets
s3 = datasets.filesystems.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# saves encoded_dataset to your s3 bucket
encoded_dataset.save_to_disk('s3://my-private-datasets/imdb/train', fs=s3)
```

## How to create Document Retreval System

Document Retreval System is a specific example from dataset which is used in NLP Task such as Question & Answer System. This document will show you how to build an index search for your dataset that will allow you to search items from your Dataset.

**FAISS(Facebook AI Similarity Search):** Dataset has mechanism where we can do similarity search on Dataset based on Embedding, here first convert long passages into a single a embeddings which can be later used as retreival system. After Converting passages into embeddings, we would convert Question into Embedding. Now with FAISS Indexing mechanism of Dataset we can compare the two embedding and get the most similar passage from the dataset.

![Semantic Search](./semanticSearch.png) 
Credits: https://huggingface.co

To do that we have One such Model named DPR  (Dense Passage Retrieval). Example taken from Huggingface Dataset Documentation. Feel free to use any other model like from sentence-transformers,etc.

**Step 1: Load the Context Encoder Model & Tokenizer.**
```python
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
torch.set_grad_enabled(False)
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
```



**Step 2: Load the Dataset and get embedding** 

As we mentioned earlier, we’d like to represent each entry as a single vector.
```python
from datasets import load_dataset
ds = load_dataset('crime_and_punish', split='train[:100]')
ds_with_embeddings = ds.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["line"], return_tensors="pt"))[0][0].numpy()})
```
 

**Step 3: Add embedding to FAISS Search Index.**

Now Using FAISS for efficient similarity search we will Create the index with ```datasets.Dataset.add_faiss_index()```
 ```python
ds_with_embeddings.add_faiss_index(column='embeddings')
 ```

**Step 4: Search Query using the FAISS Search Index**

 Now you can query your dataset with the embeddings index. Load the DPR Question Encoder, and search for a question with ```datasets.Dataset.get_nearest_examples()```

 ```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

question = "Is it serious ?"
question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding, k=10)
retrieved_examples["line"][0]
 ```

**Step 5: Save FAISS Index**

 We can save the index on disk with ```datasets.Dataset.save_faiss_index()```
 ```python
ds_with_embeddings.save_faiss_index('embeddings', 'my_index.faiss')
 ```

**Step 6: Load the FAISS Index**

 Now to load the Index from disk can be done ``` datasets.Dataset.load_faiss_index()```

 ```python
ds = load_dataset('crime_and_punish', split='train[:100]')
ds.load_faiss_index('embeddings', 'my_index.faiss')
 ```
**Elastic Search:** Unlike FAISS, ElasticSearch retrieves documents based on exact matches in the search context.

Please see the [Installation & Configuration Guide for more information](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html)

Elastic Search is pretty much similar to the FAISS Search. Example taken from Standard Huggingface Dataset Documentation.

**Step 1: Load the the Dataset**

```python
from datasets import load_dataset
squad = load_dataset('squad', split='validation')
```
**Step 2: Add Elastic Search to Dataset**
```python
squad.add_elasticsearch_index("context", host="localhost", port="9200", es_index_name="hf_squad_val_context")
```
**Step 3: Execute the query Search**

```python
query = "machine"
scores, retrieved_examples = squad.get_nearest_examples("context", query, k=10)
retrieved_examples["title"][0]
```

**Step 4: Load the Elastic Search Index if Name is provided**

```python
from datasets import load_dataset
squad = load_dataset('squad', split='validation')
squad.load_elasticsearch_index("context", host="localhost", port="9200", es_index_name="hf_squad_val_context")
query = "machine"
scores, retrieved_examples = squad.get_nearest_examples("context", query, k=10)
```



