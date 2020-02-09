# Word2Vec with transactional data
Developing product feature embedding with the word2vec algorithm for recommendation system. 


## Installation
Firstly, clone the repository to a folder of your choice. 

Next install the necessary libraries

1. Create an environment with a specific version of Python
	```
   conda create -n word2vec python=3.6
   activate word2vec
   ```

2. Install libraries from requirements.txt using conda. For windows, run the following in the terminal:
	```
   for /f %i in (requirements.txt) do conda install --yes %i
   ```

Next just run the analysis through the jupyter notebook and you're on your way to developing your own product embeddings!

## Data Source
[E-Commerce Data](https://www.kaggle.com/carrie1/ecommerce-data)


## Final product embedding for a subset of products
![T-sne of product embeddings](/images/t-sne.png)
