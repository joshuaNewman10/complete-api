# Complete-AI Model Performance Visualization Blog Post
This module is for my model performance visualization blog post (link)

This README describes the steps for running the code referenced in the post

Please follow the project README installation instructions before trying to run this code

## MNIST Embedding Experiment
This experiment trains a simple MNIST model and produces some embedding visualizations. This 

### Run Experiment
RUN `python mnist_performance_visualization_experiment.py --download_directory /tmp/images`

### Run Tensorflow Embedding Projector
To run the Embedding Projector you need to do two things. 

First run the provided script `tf_embedding_proj.py` You can find the vectors.tsv, metadata.json, and sprite.png in the training run directory 
RUN `complete-ai/scripts/tf_embedding_proj.py --log_dir /tmp/longs --vectors_file_path vectors.tsv --metadata_file_path metadata.json --sprite_file_path sprite.png`

Now start up tensorboard 
`tensorboard /tmp/longs`

Now, navigate to http://localhost:6006

## Twitter Text Sentiment LSTM Embedding Experiment
This experiment trains an LSTM model and produces some embedding visualizations

Before running this experiment you will need to download the Kaggle Twitter Sentiment Dataset: https://www.kaggle.com/kazanova/sentiment140

You will then need to convert this csv file to a jsonl file via the script I have provided twitter_sentiment_csv_to_json.py

RUN `python complete-ai/scripts/twitter_sentiment_csv_to_json.py --in_file_path tweets.csv --out_file_path tweets.jsonl`

### Run Experiment
RUN `python text_sentiment_lstm_embedding_experiment.py --embedding_file_path glove.twitter.27B.25d.txt --training_data_file_path twitter_sentiment.jsonl`

### Run Tensorflow Embedding Projector
To run the Embedding Projector you need to do two things. 

First run the provided script `tf_embedding_proj.py` You can find the vectors.tsv, metadata.json, and sprite.png in the training run directory 
RUN `complete-ai/scripts/tf_embedding_proj.py --log_dir /tmp/longs --vectors_file_path vectors.tsv --metadata_file_path metadata.json`

Now start up tensorboard 
`tensorboard /tmp/longs`

Now, navigate to http://localhost:6006


## Twitter Text Sentiment Hierarchical RNN Embedding Experiment
This experiment trains an Hierarchical RNN model and produces some embedding visualizations.

Before running this experiment you will need to download the Kaggle Twitter Sentiment Dataset: https://www.kaggle.com/kazanova/sentiment140

You will then need to convert this csv file to a jsonl file via the script I have provided twitter_sentiment_csv_to_json.py

RUN `python complete-ai/scripts/twitter_sentiment_csv_to_json.py --in_file_path tweets.csv --out_file_path tweets.jsonl`

### Run Experiment
RUN `python text_sentiment_hiearchal_rnn_embedding_experiment.py --training_data_file_path twitter_sentiment.jsonl`

### Run Tensorflow Embedding Projector
To run the Embedding Projector you need to do two things. 

First run the provided script `tf_embedding_proj.py` You can find the vectors.tsv, metadata.json, and sprite.png in the training run directory 
RUN `complete-ai/scripts/tf_embedding_proj.py --log_dir /tmp/longs --vectors_file_path vectors.tsv --metadata_file_path metadata.json`

Now start up tensorboard 
`tensorboard /tmp/longs`

Now, navigate to http://localhost:6006ai/scripts/tf_embedding_proj.py --log_dir /tmp/longs --vectors_file_path vectors.tsv --metadata_file_path metadata.json --sprite_file_path sprite.png`