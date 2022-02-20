# Forget Complex Traditional Approaches to handle NLP Datasets, HuggingFace Dataset Library is your saviour!

## Authors
**Nabarun Barua**     

[Git](https://github.com/nabarunbaruaAIML)/ [LinkedIn](https://www.linkedin.com/in/nabarun-barua-aiml-engineer/)/ [Towardsdatascience](https://medium.com/@nabarun.barua)

**Arjun Kumbakkara** 

[Git](https://github.com/arjunKumbakkara)/ [LinkedIn](https://www.linkedin.com/in/arjunkumbakkara/)/ [Towardsdatascience](https://medium.com/@arjunkumbakkara)


In the world of Data Science and Artificial Intelligence, Data plays an important role which is used either in training or to get insights. Anyone who has use either Tensorflow or Pytorch knows about Tensorflow/Pytorch Dataset Library or knows Pandas for Insights.

Now the important question to ask why do we need HuggingFace Dataset Library at all?

Answer to it is in four parts.

1. Under the hood HuggingFace Dataset Library runs on [Apache Arrow](https://arrow.apache.org/) memory format and [pyarrow library](https://arrow.apache.org/docs/python/index.html) because of which Data Loading & Processing is Lighting Fast. Dataset Library Treat each dataset as a memory-mapped file that helps mapping between RAM & File Storage which allows library to access and process the dataset without needing load dataset fully into the memory. For more information on memory-mapped file read this [link](https://en.wikipedia.org/wiki/Memory-mapped_file). This also means less amount of RAM need to process the dataset.

2. HuggingFace Dataset Library also support different types of Data format to be loaded into memory. Example CSV, TSV, Text Files, JSON & Pickled DataFrame.

3. And if need arise to work in DataFrame then simple property change in the Dataset makes it work as a DataFrame and all the function of DataFrame works here. When work of DataFrame is finished then we can simply reset the property back to Dataset. Under the hood, Memory-mapped is working which makes overall memory consumption much more efficient.

4. It also supports the Streaming Data or if Dataset is very Huge then in that condition we can load the data in the form of Streaming Dataset which is little different from normal Dataset i.e. Iteratorable Dataset.

Now that we know why would like to use HuggingFace Dataset Library. We would study about the library in detail.

## Loading Dataset

First step starts from loading the dataset. We can load Dataset in Multiple ways which are as follows:

### Offline Dataset File:
We can load the Dataset simply using a Function named load_dataset. As mentioned above we can load different format.

**Example CSV**
```python
from datasets import load_dataset

dataset1 =  load_dataset('csv', data_files= 'location/file1.csv')
```

**Example JSON**

Simple JSON

{"a": 1, "b": 2.0, "c": "foo", "d": false}

```python
from datasets import load_dataset

dataset2 =  load_dataset('json', data_files= 'location/file2.json')
```

In case of nested JSON

{"version": "0.1.0",
    "data": [{"a": 1, "b": 2.0, "c": "foo", "d": false},
            {"a": 4, "b": -5.5, "c": null, "d": true}]
}
```python
from datasets import load_dataset

dataset3 =  load_dataset('json', data_files= 'location/file3.json', field= 'data')
```
**Example Text File**
```python
from datasets import load_dataset

dataset4 =  load_dataset('text', data_files= 'location/file4.txt')
```

**Example Parquet**
```python
from datasets import load_dataset

dataset5 =  load_dataset('parquet', data_files= 'location/file5.parquet')
```
Note:Parquet is just another file storage format .An open source file format built to handle flat columnar storage data formats.

**Example Loading Zip/TAR or common compressed file for JSON or anyother format**

```python
from datasets import load_dataset

dataset6 =  load_dataset('json', data_files= 'location/file6.json.gz', field= 'data')
```

Now if we don't give split in above example then data will be loaded by default in Train split. To have train-test split while loading the Dataset we can do with following changes: 

```python
from datasets import load_dataset

DataFile = {'train':['location/file7.csv','location/file8.csv','location/file9.csv'],'test':'location/file10.csv'}

dataset7 =  load_dataset('csv', data_files= DataFile )
```

Simialarly for other file format we can use above technique. Now if by chance we don't have dataset file with train & test then we can split dataset following steps

```python
from datasets import load_dataset

dataset8 =  load_dataset('csv', data_files= 'location/file8.csv')
dataset8 = dataset8['train']
dataset8 = dataset8.train_test_split(test_size=0.1)
```

### Remote Dataset or Loading Dataset from url

In HuggingFace Dataset Library, we can also load remote dataset stored in a server as a local dataset. As a Data Scientist in real world scenario most of time we would be loading data from a remote server. We simpily have to give url instead local path.

**Example**

```python
url = "https://https://github.com/crux82/squad-it/raw/master/"

data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}

dataset = load_dataset("json", data_files=data_files, field="data")

```

Similarly approach can be taken for other data type.

### Loading Streaming Data

Now a days Data are available in GB's and TB's because of that it's not possible to load the complete dataset in the system. For such scenario's we load dataset in stream format which is bit different. Normally load_dataset returns a data type of  Dataset but in Streaming format it returns a data type of IterableDataset. We learn more about streaming Data later in detail.

We can load any dataset in streaming format by simply passing property streaming=True

**Example**
```python
url = "https://https://github.com/crux82/squad-it/raw/master/"

data_files = url + "SQuAD_it-train.json.gz"

dataset = load_dataset("json", data_files=data_files, field="data",split="train",streaming=True)

```

### In-Memory Data

HuggingFace Dataset Library also allows to create a dataset from a Dictionary & DataFrame.

**Example Dataset from Dictionary**

```python
from datasets import Dataset

my_dict = {'title':['Macbeth','Tempest','Julius Caesar'],'character':['King Duncan of Scotland','Prospero','Brutus']}

dataset = Dataset.from_dict(my_dict)
```

**Example Dataset from DataFrame**

```python
from datasets import Dataset
import pandas as pd

my_dict = {'title':['Macbeth','Tempest','Julius Caesar'],'character':['King Duncan of Scotland','Prospero','Brutus']}

df = pd.DataFrame(my_dict)

dataset = Dataset.from_pandas(df)
```

## Preprocessing Dataset
Now we know how to load the Dataset, it's time to learn how to process the raw dataset. In general we may not get a clean Dataset and we have to process the dataset by cleaning and bring it to a level where we can sent it to model.


**Select Record in Dataset**

In the below example we're taking first 1000 entries from the Dataset and returning data type is Dataset

We can select the record by following example

```python
sample = dataset['train'].select(range(1000))
```

Now similarly if we want to see the actual data then we can do the same by simple change which will return data type Dictionary.

```python
sample = dataset['train'][0:1000]
```

**Shuffle Dataset**

Now let's say we want to select dataset but want data point to be shuffled before selecting Dataset or you want to shuffle complete dataset.
```python
sample = dataset['train'].shuffle(seed=34).select(range(1000))
```

**To get unique entries from the dataset column**

Requirement may araise where a Data Scientist may want to know what are the unique entries available in the column of a dataset.

with the below example we will get a list of unique entries

```python
list = dataset['train'].unique("Title")
```

**Sort Dataset**

Here Column must have Numerical values i.e. Column must be Numpy compatible, as columns values will be sorted according to their numerical values.

```python
# It will sort the column in descending Order
sorted_dataset = dataset.sort('label',reverse=True)
```
**Filter Dataset**

There may be a condition when a Data Scientist may want to get a specific type of Data from Dataset. So HuggingFace Dataset Library has a Function named Filter which extract data points which matches it's filter condition.

In the below example if there is a Data point in Dataset for Column "Character_Name" having value none, in that condition that Data point will be skipped.

```python
dataset = dataset.filter(lambda x: x['Character_Name'] is not None)
```

**Renaming the Column**

HuggingFace Dataset Library allows you to rename the column of the Dataset. We can understand by the following example, here pass the Actual Column Name i.e. 'Title' and the Column Name to be renamed i.e. 'Novel'.
```python
dataset = dataset.rename_column("Title", "Novel")
```

**Removing the Column**

Similar to Rename, their could be a scenario where Data Scientist need to remove columns from the Dataset. We can understand by the following example, here we pass the list of Columns which we want to remove from the Dataset.
```python
dataset = dataset.remove_columns(['ID', 'Texts'])
```

**Cast Data Type for the Column**

Their could be a scenario where Data Scientist may have to change the Feature type of a column then in that scenario we can cast the column.

```python
dataset.features
```
**Output**
```bash
{'sentence1': Value(dtype='string', id=None),
'sentence2': Value(dtype='string', id=None),
'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
'idx': Value(dtype='int32', id=None)}
```

As seen , we cast Data Type for the Column labels to text binaris such as 'positive '  and 'negative' 

```python
from datasets import ClassLabel, Value
new_features = dataset.features.copy()
new_features["label"] = ClassLabel(names=['negative', 'positive'])
new_features["idx"] = Value('int64')
dataset = dataset.cast(new_features)
dataset.features
```
**Output**
```bash
{'sentence1': Value(dtype='string', id=None),
'sentence2': Value(dtype='string', id=None),
'label': ClassLabel(num_classes=2, names=['negative', 'positive'], names_file=None, id=None),
'idx': Value(dtype='int64', id=None)}
```
**Flatten**

Sometimes a column can be a nested structure of several types and need is to extract the subfields into their own separate columns which can be done with Function Flatten.

```python
from datasets import load_dataset
dataset = load_dataset('squad', split='train')
dataset.features
```
**Output**
```bash
{'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
'context': Value(dtype='string', id=None),
'id': Value(dtype='string', id=None),
'question': Value(dtype='string', id=None),
'title': Value(dtype='string', id=None)}
```
The answers field contains two subfields: text and answer_start. Flatten them with datasets.Dataset.flatten():

```python
flat_dataset = dataset.flatten()
flat_dataset
```
**Output**
```bash
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
 num_rows: 87599
})
```

**Map() method**

This is a special method which allows to update a Column/Coulmns or create a new Column/Columns in Dataset.

example of updating a column. Below we are lowering title and updating the column.

```python
dataset = dataset.map(lambda x: x['Title'].lower())
```

example of creating a New Column. Below we are counting word in Column Title and saving in new Column.

```python
def compute_length(example):
    return {"length": len(example["Title"].split())}
dataset = dataset.map(compute_length)
```

Now in the above example, we are processing the values one at a time which we are not using full potential of the Library. Here we can do the same stuff in Batches by simiply adding a property batched=True (the batch size is configurable but defaults to 1,000).
```python
def compute_length(example):
    return {"length": len(example["Title"].split())}
dataset = dataset.map(compute_length,batched=True)
```
Use of this Function can be best seen when using with Fast Tokenizer.

Below example taken from HuggingFace.co which shows relative comparison for Fast & slow tokenizer with batch & without batch on a dataset.

```bash
Options	            Fast tokenizer	Slow tokenizer
batched=True	    10.8s	        4min41s
batched=False	    59.2s	        5min3s
```
By above example we can see that if batched is true then it's 30 times faster in executing the function.

This Fucntion also support parallelism i.e. property num_proc. But do remember using num_proc to speed up your processing is usually a great idea, as long as the function you are using is not already doing some kind of multiprocessing of its own. In Below example we have used multiprocessing with Fast tokenizer but that is not advisable since Fast tokenizer already works internally on parallelism. For better understanding please go through the HuggingFace.co. Below example taken from their course which shows relative comparison for Fast & slow tokenizer with num_proc,batch & without batch,num_proc on a dataset.

```bash
Options	                            Fast tokenizer	Slow tokenizer
batched=True	                    10.8s	        4min41s
batched=False	                    59.2s	        5min3s
batched=True, num_proc=8	    6.52s	        41.3s
batched=False, num_proc=8	    9.49s	        45.2s

```

Above are the major functions which are mostly used for day to day work.

## Dataset to DataFrame

There could be a time when a Data Scientist would wish to see data in DataFrame and do some EDA on it. If Data Scientist is using HuggingFace Dataset Library then he/she can simply do it by setting this Function **set_format** to Pandas. Example
```python
dataset.set_format('pandas')
```
This function only changes the output format of the dataset, so you can easily switch to another format without affecting the underlying data format, which is Apache Arrow. The formatting is done in place.

Now when we access elements of the dataset we get a pandas.DataFrame instead of a dictionary:

```python
DF = dataset['train'][:3]
```
Here all the Pandas functions will work as expected. And when work on the dataset is complete then we can reset it to **arrow** format. Example
```python
dataset.reset_format()
```

## Streaming Dataset

Above we have spoken about Streaming Dataset. Since Streaming Dataset are iteratorable Dataset therefore some processing functions are different here than normal datasets which will be explained.

**Map() method**

First change is how mapping function works if it's used without batched = True then in Stream Dataset outputs are returned one by one whereas Normal Dataset complete Dataset is returned. Therefore for optimal use of streaming dataset, use Batched=True which will take batch data from streaming dataset. By default batch size is 1000 but configurable.

**Shuffle() method**

Unlike Shuffle in Normal Dataset which shuffles the entire dataset, here shuffle only shuffle the element of the batch that is predefined buffer_size.


**Take() & Skip() method**

In Streaming Dataset select() method doesn't work, instead we have two method take() & skip() method which is similar to select() method.

Let's say Buffer size is 1000 and we accessed one value from dataset then buffer is filled or replaced with 1001 place value.

For example if we want to take first 10 records:

```python
dataset = streamed_dataset.take(10)
```
Similarly if we want to create training and validation splits from a streamed dataset
```python
# Skip the first 1,000 examples and include the rest in the training set
train_dataset = streamed_dataset.skip(1000)
# Take the first 1,000 examples for the validation set
validation_dataset = streamed_dataset.take(1000)
```

There are lot's of things to be told for this Library.Hence we have decided to make this a series based on usage level such as intermediate and advanced.
Please watch out for the Advanced version of the same repo. You can click on 'Watch' and stay upto date using notifications.

#### Advanced Concepts on Level Advanced 
##### 1. More on Parallized File Systems 
##### 2. Cloud based Storages , Retrieval and all pertinent operations.
##### 3. Data Collation : Collate large chunks of data into processable units 

If you like this Blog please show your love and give us a thumbs up , star us  and if not please do give us a feedback in the comment section.
Here's hoping that you wil have fun with the library! 


##### For Collaboration , Help and Learning things together  - 
#### Join our Discord Server :  https://discord.gg/Z7Kx96CYGJ

#####  GodSpeed! 
