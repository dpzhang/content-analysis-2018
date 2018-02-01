#Special module written for this class
#This provides access to data and to helper functions from previous weeks
#Make sure you update it before starting this notebook
import lucem_illud #pip install -U git+git://github.com/Computational-Content-Analysis-2018/lucem_illud.git
#All these packages need to be installed from pip
import gensim#For word2vec, etc
import requests #For downloading our datasets
import nltk #For stop words and stemmers
import numpy as np #For arrays
import pandas #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import sklearn.metrics.pairwise #For cosine similarity
import sklearn.manifold #For T-SNE
import sklearn.decomposition #For PCA
#gensim uses a couple of deprecated features
#we can't do anything about them so lets ignore them 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os #For looking through files
import os.path #For managing file paths


################################################################################
# Construct cells immediately below this that build a word2vec model with your corpus. 
# Interrogate word relationships in the resulting space, including 
    # estimating 90% confidence intervals for specific word cosine distances of interest. 
    # Plot a subset of your words. What do these word relationships reveal about the social and cultural game underlying your corpus? 
    # What was surprising--what violated your prior understanding of the corpus? What was expected--what confirmed your knowledge about this domain?
################################################################################

# My corpus contains 54 translated English novels from China, Korea, and U.S. 
    # The majority of the translated Chinese novels could be categorized to a genre called
        # eastern fantasy as it involve topics such as Kung Fu, Sage, Chinese 
        # gods, etc.  
    # The majority of the translated Korean novels could be only categorized as
        # fantasy. Korean novels is heavily influenced by its western counterparts.
        # They first mimic the western novels, and gradually develope its
        # own characteristics and style.
    # The U.S. novels are composed by western online readers, who are inspired by 
        #these Chinese and Korean novels and started their own literary attempts

bookDF = pandas.read_csv('bookDF.csv', sep = ';', usecols = list(range(1,5)))

# To remove stop words and stem
    # first tokenize sentences to capture a "continuous bag of words (CBOW)"
    # next normalize sentences 
bookDF['tokenized_sents'] = bookDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
bookDF['normalized_sents'] = bookDF['tokenized_sents'].apply(lambda x: 
             [lucem_illud.normalizeTokens(s, 
                                          stopwordLst = lucem_illud.stop_words_basic, 
                                          stemmer = None) for s in x])

# Use my corpus to biild a word2vec model
bookW2V = gensim.models.word2vec.Word2Vec(bookDF['normalized_sents'].sum())

############### to do some quick exploratory analysis
    # since the majority of the novels are eastern fantasies
    # lets check vector 'punch'
print("A {} dimesional vector:".format(bookW2V['punch'].shape[0]))
bookW2V['punch']

    # Find similar vectors by cosine similarity
# because many scenes in these novels involves fighting so I tried some words 
# and to assess their most similiar counterparts
    # expectedly, looking at similar words of 'fight', it would be "win", and to win means to "survive"
    # similarly, looking at similar words of 'lose', it would be "suffer", "fail" and "suicide"
    # somehow, from those similar words to our words of interest, we could get a sense of what Chinese culture, or Eastern Asian Culture, is like by assessing thoses generated returns. 
wordTestLst = ['kick', 'punch', 'strong', 'fight', 'lose']
[ [bookW2V.most_similar(word) for word in wordTestLst] ]
# interesting, when I tried to assess some emotional words such as "damn", it
# would return a list of derogratory words and vulgarity
bookW2V.most_similar('damn')
# Try directly compare two words by their cosine similarity
def cos_difference(embedding,word1,word2):
    return sklearn.metrics.pairwise.cosine_similarity(embedding[word1].reshape(1,-1),embedding[word2].reshape(1,-1))
cos_difference(bookW2V, 'win', 'lose')
cos_difference(bookW2V, 'haha', 'hehe')

# try to find words that least matches the others within a word set through cosine similarity
# it turns out "blood" is not similiar witht he rest, which sort of makes senbse. 
# it is the only noun from the rest of the verbs and the other words are happened
# during the fight and involves movement, but "blood" is more like of a consequence. 
bookW2V.doesnt_match(['kick', 'punch', 'fight', 'blood', 'dodge'])

# Find which word best matches the result of a semantic equation
# It is very interesting results: love + women - beautiful = friends, love + men - handsome = bandit (thats harsh)
# also, 
bookW2V.most_similar(positive=['love', 'women'], negative = ['beautiful'])
bookW2V.most_similar(positive=['love', 'men'], negative = ['handsome'])
# also it could be seen the return of the equatrion: women - good = men. this is very unexpected and maybe because most of villans in the books are all males?
bookW2V.most_similar(positive=['women'], negative = ['good'])




### Establishing Credible or Confidence Intervals
# Because my corpus is large, a subsampling approach would be used.To easily implement repeititive process, a function would be constructed to simplify the repetitive process. 
# This function will randomly partitions the corpus into non-overlapping samples, then estimates the word-embedding models on these subsets and calculates confidence intervals as a function of the empirical distribution of distance or projection statistics and number of texts in the subsample.
    

def subsamplingCI(bookDF, n_samples, word1, word2):
    sample_indices = np.random.randint(0,n_samples,(len(bookDF),))
    while len(np.unique(sample_indices)) != n_samples:
        sample_indices = np.random.randint(0,n_samples,(len(bookDF),))  
    
    s_k =np.array([])
    tau_k=np.array([])
    
    for i in range(n_samples):
        sample_w2v = gensim.models.word2vec.Word2Vec(bookDF[sample_indices == i]['normalized_sents'].sum())
        try:
            #Need to use words present in most samples
            s_k = np.append(s_k, cos_difference(sample_w2v, word1, word2)[0,0])
        except KeyError:
            pass
        else:
            tau_k = np.append(tau_k, len(bookDF[sample_indices == i]))
    
    print("Averaged Cosine Difference in each subcorpora")
    print(s_k)
    print("Number of obs assigned to each subcorpora")
    print(tau_k)
    
    tau = tau_k.sum()
    s = s_k.mean()
    B_k = np.sqrt(tau_k) * s_k-s_k.mean() 
    print("The 90% confidence interval for the cosine distance between {} and {} is:\n".format(word1, word2),
          s-B_k[-2]/np.sqrt(tau), s-B_k[1]/np.sqrt(tau))
    return s-B_k[-2]/np.sqrt(tau), s-B_k[1]/np.sqrt(tau)

### estimating 90% confidence intervals for specific word cosine distances of interest.
# I am interested to observe some gender differences: using the code below, it seems in eastern novels, woman are more likely to be associated with good-looking and beautiful, but man are more likely to be associated with smart and intelligent. 
# However, interestingly, when I changed man and woman to their young-aged version "boy" and "girl", this finding has completely become different. Girls are more associated with smart than boys. 
subsamplingCI(bookDF, 20, 'man', 'handsome')
subsamplingCI(bookDF, 20, 'man', 'smart')
subsamplingCI(bookDF, 20, 'woman', 'pretty')
subsamplingCI(bookDF, 20, 'woman', 'smart')

subsamplingCI(bookDF, 20, 'boy', 'handsome')
subsamplingCI(bookDF, 20, 'boy', 'smart')
subsamplingCI(bookDF, 20, 'girl', 'pretty')
subsamplingCI(bookDF, 20, 'girl', 'smart')


### Visualization
# Selecting a subset we want to plot
def createPlot(w2vObject, numWords, n_components):
    targetWords = w2vObject.wv.index2word[:numWords]
    # extract their vectors and create our own smaller matrix that preserved the distances from the original
    wordsSubMatrix = []
    for word in targetWords:
        wordsSubMatrix.append(w2vObject[word])
    wordsSubMatrix = np.array(wordsSubMatrix)
    wordsSubMatrix
    # use PCA to reduce the dimesions (e.g., to 50), and T-SNE to project them down to the two we will visualize.
    pcaWords = sklearn.decomposition.PCA(n_components).fit(wordsSubMatrix)
    reducedPCA_data = pcaWords.transform(wordsSubMatrix)
    #T-SNE is theoretically better, but you should experiment
    tsneWords = sklearn.manifold.TSNE(n_components = 2, early_exaggeration = 25).fit_transform(reducedPCA_data)
    # plot
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    plt.scatter(tsneWords[:, 0], tsneWords[:, 1], alpha = 0)#Making the points invisible 
    for i, word in enumerate(targetWords):
        ax.annotate(word, 
                    (tsneWords[:, 0][i],tsneWords[:, 1][i]), 
                    size =  20 * (numWords - i) / numWords, 
                    alpha = .8 * (numWords - i) / numWords + .2)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# To make sure that our model is correct, looking at the training loss of the model: it turns out that the loss is 2116754
bookW2V_loss = gensim.models.word2vec.Word2Vec(size = 100, #dimensions
                                                      alpha=0.025,
                                                      window=5,
                                                      min_count=5,
                                                      hs=0,  #hierarchical softmax toggle
                                                      compute_loss = True,
                                                     )
bookW2V_loss.build_vocab(bookDF['normalized_sents'].sum())
bookW2V_loss.train(bookDF['normalized_sents'].sum(), 
                     total_examples=bookW2V.corpus_count, 
                     epochs=1, #This the running_training_loss is a total so we have to do 1 epoch at a time
                    )
#Using a list so we can capture every epoch
losses = [bookW2V_loss.running_training_loss]
losses[0]

# Now we have the training loss and can optimize training to minimize it.
for i in range(19):
    bookW2V_loss.train(bookDF['normalized_sents'].sum(), 
                     total_examples=bookW2V.corpus_count, 
                     epochs=1,
                             )
    losses.append(bookW2V_loss.running_training_loss)
    print("Done epoch {}".format(i + 2), end = '\r')

# plot the loss vs epoch
lossesDF = pandas.DataFrame({'loss' : losses, 'epoch' : range(len(losses))})
lossesDF.plot(y = 'loss', x = 'epoch', logy=False, figsize=(15, 7))
plt.show()

# in order to avoid overfitting, choose the number of iter to be 8
# after knowing the optimal iter or epochs is 9, test the number of dimensions
losses_dims=[]

for d in [50,100,150,200,250,300,350,400,450,500, 550, 600, 650, 700, 750]:
    bookW2V_loss_dims = gensim.models.word2vec.Word2Vec(size = d, #dimensions
                                                        alpha=0.025,
                                                        window=5,
                                                        min_count=5,
                                                        hs=0,  #hierarchical softmax toggle
                                                        compute_loss = True,
                                                        iter = 9
                                                        )  
    bookW2V_loss_dims.build_vocab(bookDF['normalized_sents'].sum())
    bookW2V_loss_dims.train(bookDF['normalized_sents'].sum(), 
                     total_examples=bookW2V.corpus_count, 
                     epochs=7, #This the running_training_loss is a total so we have to do 1 epoch at a time
                    )
    bookW2V_loss_dims.train(bookDF['normalized_sents'].sum(), 
                     total_examples=bookW2V.corpus_count, 
                     epochs=1, #This the running_training_loss is a total so we have to do 1 epoch at a time
                    )
    
    losses_dims.append(bookW2V_loss_dims.running_training_loss/(10+d*10))
# plot
losses_dimsDF = pandas.DataFrame({'loss' : losses_dims, 'dimensions' : [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750]})
losses_dimsDF.plot(y = 'loss', x = 'dimensions', logy=False, figsize=(15, 7))
plt.show()




################################################################################
# Construct cells immediately below this that build a doc2vec model with your corpus. 
    # Interrogate document and word relationships in the resulting space. 
    # Construct a heatmap that plots the distances between a subset of your documents against each other, 
        #and against a set of informative words. 
    # Find distances between every document in your corpus and a word or query of interest. 
    # What do these doc-doc proximities reveal about your corpus? 
    # What do these word-doc proximities highlight? Demonstrate and document one reasonable way 
        # to select a defensible subset of query-relevant documents for subsequent analysis.
################################################################################
# load the df
bookDF = pandas.read_csv('bookDF.csv', sep = ';', usecols = list(range(1,5)))
bookDF.loc[1:5]
# create a numeric index
origin = []
for element in bookDF['country']:
    if element  == 'China':
        origin.append(0)
    if element == 'Korea':
        origin.append(1)
    if element == 'US':
        origin.append(2)
bookDF['origin'] = origin
# change book name a bit
bookDF['name'].loc[13] = 'Invincible Novel'
bookDF['name'].loc[42] = 'Overgeared Novel'
bookDF['name'].loc[45] = 'Breakers Novel'



# load these as documents into Word2Vec, but first we need to normalize and pick some tags
keywords = ['fight', 'master', 'protect', 'technique', 'god', 'sage', 'training', 'power', 'realm', 'karma', 'slay', 'gorgeous', 'beauty', 'goddess']
bookDF['tokenized_words'] = bookDF['text'].apply(lambda x: nltk.word_tokenize(x))
bookDF['normalized_words'] = bookDF['tokenized_words'].apply(lambda x: lucem_illud.normalizeTokens(x, stopwordLst = lucem_illud.stop_words_basic, stemmer = None)) 

# choose country and book name
taggedDocs = []
for index, row in bookDF.iterrows():
    #Just doing a simple keyword assignment
    docKeywords = [s for s in keywords if s in row['normalized_words']]
    docKeywords.append(row['origin'])
    docKeywords.append(row['name']) #This lets us extract individual documnets since doi's are unique
    taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words = row['normalized_words'], tags = docKeywords))
bookDF['TaggedText'] = taggedDocs

# train a Doc2Vec model:
bookD2V = gensim.models.doc2vec.Doc2Vec(bookDF['TaggedText'], size = 100) #Limiting to 100 dimensions

## Exploratory Text Analysis 
bookD2V.docvecs[0]
# access word
bookD2V['powerful']

# use the most_similar command to perform simple semantic equations:
bookD2V.most_similar(positive = ['kick','punch'], negative = ['mercy'], topn = 1)
bookD2V.most_similar(positive = ['technique','fights'], negative = ['fight'], topn = 1)
bookD2V.most_similar(positive = ['woman','beautiful'], negative = ['man'], topn = 1)
bookD2V.most_similar(positive = ['test','pen'], negative = ['smart'], topn = 1)                                                          
bookD2V.docvecs.most_similar([ bookD2V['sage'] ], topn=5 )


# calculate the distance between a word and documents in the dataset
bookD2V.docvecs.most_similar([ bookD2V['fight'] ], topn=5 )
# find words most similar to this document:
bookD2V.most_similar( [ bookD2V.docvecs['Perfect World'] ], topn=5) 
# look for documents most like a query composed of multiple words:
bookD2V.docvecs.most_similar([ bookD2V['monster']+bookD2V['beast']+bookD2V['orc']], topn=5 )

# plot some words and documents against one another with a heatmap:
def doc_doc(bookD2V, keywords):
    heatmapMatrix = []
    for tagOuter in keywords:
        column = []
        tagVec = bookD2V.docvecs[tagOuter].reshape(1, -1)
        for tagInner in keywords:
            column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, bookD2V.docvecs[tagInner].reshape(1, -1))[0][0])
        heatmapMatrix.append(column)
    heatmapMatrix = np.array(heatmapMatrix)
    fig, ax = plt.subplots()
    hmap = ax.pcolor(heatmapMatrix, cmap='terrain')
    cbar = plt.colorbar(hmap)
    
    cbar.set_label('cosine similarity', rotation=270)
    a = ax.set_xticks(np.arange(heatmapMatrix.shape[1]) + 0.5, minor=False)
    a = ax.set_yticks(np.arange(heatmapMatrix.shape[0]) + 0.5, minor=False)
    
    a = ax.set_xticklabels(keywords, minor=False, rotation=270)
    a = ax.set_yticklabels(keywords, minor=False)


# Now let's look at a heatmap of similarities between the first ten documents in the corpus:
def doc_doc(bookDF, bookD2V, numBook):
    targetDocs = bookDF['name'][:10]
    
    heatmapMatrixD = []
    
    for tagOuter in targetDocs:
        column = []
        tagVec = bookD2V.docvecs[tagOuter].reshape(1, -1)
        for tagInner in targetDocs:
            column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, bookD2V.docvecs[tagInner].reshape(1, -1))[0][0])
        heatmapMatrixD.append(column)
    heatmapMatrixD = np.array(heatmapMatrixD)
    # plot
    fig, ax = plt.subplots()
    hmap = ax.pcolor(heatmapMatrixD, cmap='terrain')
    cbar = plt.colorbar(hmap)
    
    cbar.set_label('cosine similarity', rotation=270)
    a = ax.set_xticks(np.arange(heatmapMatrixD.shape[1]) + 0.5, minor=False)
    a = ax.set_yticks(np.arange(heatmapMatrixD.shape[0]) + 0.5, minor=False)
    
    a = ax.set_xticklabels(targetDocs, minor=False, rotation=270)
    a = ax.set_yticklabels(targetDocs, minor=False)


# Now let's look at a heatmap of similarities between the first ten documents and our keywords:
def doc_words(bookD2V, keywords):
    heatmapMatrixC = []
    
    for tagOuter in targetDocs:
        column = []
        tagVec = bookD2V.docvecs[tagOuter].reshape(1, -1)
        for tagInner in keywords:
            column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, bookD2V.docvecs[tagInner].reshape(1, -1))[0][0])
        heatmapMatrixC.append(column)
    heatmapMatrixC = np.array(heatmapMatrixC)
    fig, ax = plt.subplots()
    hmap = ax.pcolor(heatmapMatrixC, cmap='terrain')
    cbar = plt.colorbar(hmap)
    
    cbar.set_label('cosine similarity', rotation=270)
    a = ax.set_xticks(np.arange(heatmapMatrixC.shape[1]) + 0.5, minor=False)
    a = ax.set_yticks(np.arange(heatmapMatrixC.shape[0]) + 0.5, minor=False)
    
    a = ax.set_xticklabels(keywords, minor=False, rotation=270)
    a = ax.set_yticklabels(targetDocs, minor=False)


# Projection
#words to create dimensions
bookDF = pandas.read_csv('bookDF.csv', sep = ';', usecols = list(range(1,5)))
bookDF['tokenized_sents'] = bookDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
bookDF['normalized_sents'] = bookDF['tokenized_sents'].apply(lambda x:          
             [lucem_illud.normalizeTokens(s,                                    
                                          stopwordLst = lucem_illud.stop_words_basic, 
                                          stemmer = None) for s in x])
bookW2V = gensim.models.word2vec.Word2Vec(bookDF['normalized_sents'].sum())  

bookTargetWords = ['strong', 'weak', 'beautiful', 'ugly', 'handsome', 'ordinary', 'amazing', 'terrible', 'rich', 'richer', 'richest', 'expensive', 'cheap', 'poor', 'poorer', 'poorest', 'precious', 'mundane', 'normal', 'brilliant']
#words we will be mapping
bookTargetWords += ['master', 'god', 'sage', 'king', 'emperor', 'dominate', 'people', 'farmer', 'technique', 'skill', 'mountain', 'river', 'star', 'universe', 'territory', 'war', 'peace', 'train', 'lesson', 'death', 'suffer', 'die', 'survive', 'faint', 'hurt', 'medicine', 'spirit', 'winner', 'loser', 'suicide', 'murder', 'relationship', 'kin', 'village', 'sect', 'clan', 'palace']

wordsSubMatrix = []
for word in bookTargetWords:
    wordsSubMatrix.append(bookW2V[word])
wordsSubMatrix = np.array(wordsSubMatrix)
wordsSubMatrix

# PCA dimension reduction
pcaWordsBook = sklearn.decomposition.PCA(n_components = 50).fit(wordsSubMatrix)
reducedPCA_dataBook = pcaWordsBook.transform(wordsSubMatrix)
#T-SNE is theoretically better, but you should experiment
tsneWordsBook = sklearn.manifold.TSNE(n_components = 2).fit_transform(reducedPCA_dataBook)

# Plot word cloud
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(tsneWordsBook[:, 0], tsneWordsBook[:, 1], alpha = 0) #Making the points invisible
for i, word in enumerate(tnytTargetWords):
    ax.annotate(word, (tsneWordsBook[:, 0][i],tsneWordsBook[:, 1][i]), size =  20 * (len(tnytTargetWords) - i) / len(tnytTargetWords))
plt.xticks(())
plt.yticks(())
plt.show()
