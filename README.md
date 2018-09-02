# TripletEmbeddingModel
Embedding method of subject-verb-object (SVO) triplet data in vector space

## Description

#### 1 - DataPrep (Prepare Data)
- **WebScrap_multi** ([code](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/1-DataPrep/WebScrap_multi.py))
Reuter news : web scraping by using `BeautifulSoup`
- **CleanTextData** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/1-DataPrep/CleanTextData.ipynb))
Some preprocessing of Reuters news data (headlines and bodies)

#### 2 - SVO
- **ExtractSVO** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/2-SVO/ExtractSVO.ipynb))
Extract *subject-verb-object* from headlines by using `stanfordcorenlp`
- **ExtractSVO_chunker** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/2-SVO/ExtractSVO_chunker.ipynb))
Extract *subject-verb-object* from headlines by using chunker trained on `CoNLL-2000`
- **MatchSVO** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/2-SVO/MatchSVO.ipynb)) 
Some preprocessing of data and get average vector for *triplet embedding model*
- **MatchSVO_multi** ([code](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/2-SVO/MatchSVO_multi.py))
get average vector for *triplet embedding model* by using `multiprocessing`

#### 3 - Word2Vec
- **Word2Vec** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/3-Word2vec/Word2Vec.ipynb)) Train skipgram model for converting each word in *subject-verb-object* into dense vectors

#### 4 - TripletEmb (Triplet Embedding Model)
- **TripletEmb_train** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/4-TripletEmb/TripletEmb_train.ipynb))
Train *triplet embedding model*
- **TripletEmb_v1_BN** ([code](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/4-TripletEmb/TripletEmb_v1_BN.py))
Triplet embedding model (refered to Ding et al.(2015))
- **TripletEmb_v1_BN_TripletLoss** ([code](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/4-TripletEmb/TripletEmb_v1_BN_TripletLoss.py))
Triplet embedding model with triplet loss function (Inspired by Schroff et al.(2015))

#### 5 - Evaluation (by using t-SNE)
- **Evaluation** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/5-Evaluation/Evaluation.ipynb))
Evaluate triplet vectors by using `bokeh` and `Multicore-TSNE`[https://github.com/DmitryUlyanov/Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
- **Testset** ([notebook](https://github.com/junhyeok-kim/TripletEmbeddingModel/blob/master/5-Evaluation/Testset.ipynb))
Generate testset for evaluation.
- **testset1.csv, testset2.csv**
You can see testsets what I made.

## Dataset
We used Reuters news data.
Due to the copyright issue, you can find news data in Reuters Archive ([https://www.reuters.com/resources/archive/us/index.html](https://www.reuters.com/resources/archive/us/index.html))

## Results



