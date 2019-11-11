from param_config import config
import _pickle as cPickle
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
from difflib import SequenceMatcher
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

spchars = re.compile('\`|\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\_|\+|\=|\\|\||\{|\[|\]|\}|\:|\;|\'|\"|\<|\,|\>|\?|\/|\.|\-')

def exact_similarity(Query, Candidate):
    Q = set(Query.split())
    C = set(Candidate.split())
    if len(Q) > 0:
        similarity = len(Q.intersection(C))*1.0/len(Q)
    else:
        similarity = -1
    if similarity<1 and similarity >= 0:
        similarity=0
    return similarity

def common_words(Query, Candidate):
    Q = set(Query.split())
    C = set(Candidate.split())
    similarity = 1.0*len(Q.intersection(C))
    return similarity

def load_stopwords(filename):
    stopwords = set()
    with open(filename, "r") as fin:
        for line in fin:
            tok = line.strip()
            stopwords.add(tok.lower())
    return(stopwords)
stopwords = load_stopwords("stopwords.txt")

def is_stopword(token):
    if token in stopwords:
        return(True)
    else:
        return(False)

def normalize(text):
    # convert text to lowercase
    text = text.lower()
    # remove special characters
    text = spchars.sub(" ", text)
    #remove digits
    text = re.sub('\d', ' ', text)
    #remove customized stopwords
    tokens=[token for token in text.split() if not is_stopword(token)]
    return " ".join(tokens)

def stem(s):
    stemmer = PorterStemmer()
    s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
    return s

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def tokens_with_pos(document):
    return " ".join([x[0] for x in nltk.pos_tag(document.split()) if x[1] in ['NN', 'JJ'] and len(x[0])>2])

def preprocess_data(data, suffix):
    print("Preprocessing: {} data normalization...".format(suffix))
    if os.path.exists(config.output_folder+"/{}_normalized.pkl".format(suffix)):
        with open(config.output_folder+"/{}_normalized.pkl".format(suffix),"rb") as f:
            train = cPickle.load(f)
    else:
        data["search_term"] = data.apply(lambda row: normalize(row["search_term"]),axis=1)
        data["product_title"] = data.apply(lambda row: normalize(row["product_title"]),axis=1)
        data["product_description"] = data.apply(lambda row: normalize(row["product_description"]),axis=1)
        data["all_text"] = data["search_term"] + " " + data["product_title"] + " " + data["product_description"]
        with open(config.output_folder+"/{}_normalized.pkl".format(suffix),"wb") as f:
            cPickle.dump(data,f,-1)

    print("Preprocessing: {} data stemming...".format(suffix))
    if os.path.exists(config.output_folder+"/{}_stemmed.pkl".format(suffix)):
        with open(config.output_folder+"/{}_stemmed.pkl".format(suffix),"rb") as f:
            data = cPickle.load(f)
    else:
        data["search_term"] = data.apply(lambda row: stem(row["search_term"]),axis=1)
        data["product_title"] = data.apply(lambda row: stem(row["product_title"]),axis=1)
        data["product_description"] = data.apply(lambda row: stem(row["product_description"]),axis=1)
        data["all_text"] = data["search_term"] + " " + data["product_title"] + " " + data["product_description"]
        with open(config.output_folder+"/{}_stemmed.pkl".format(suffix),"wb") as f:
            cPickle.dump(data,f,-1)

    print("Preprocessing: {} data - keep only noun and adjectives...".format(suffix))
    if os.path.exists(config.output_folder+"/{}_normalized_stemmed_pos.pkl".format(suffix)):
        with open(config.output_folder+"/{}_normalized_stemmed_pos.pkl".format(suffix),"rb") as f:
            data = cPickle.load(f)
    else:
        data["search_term"] = data.apply(lambda row: tokens_with_pos(row["search_term"]),axis=1)
        data["product_title"] = data.apply(lambda row: tokens_with_pos(row["product_title"]),axis=1)
        data["product_description"] = data.apply(lambda row: tokens_with_pos(row["product_description"]),axis=1)
        data["all_text"] = data["search_term"] + " " + data["product_title"] + " " + data["product_description"]

        with open(config.output_folder+"/{}_normalized_stemmed_pos.pkl".format(suffix),"wb") as f:
            cPickle.dump(data,f,-1)

    return data

def generate_features(train, test):

    print("generating feature 1")
    #feature 1: Query exactly occurs in title (order of words not important)
    train["search_term_in_title"]=train.apply(lambda row: exact_similarity(str(row['search_term']),row['product_title']),axis=1)
    test["search_term_in_title"]=test.apply(lambda row: exact_similarity(str(row['search_term']),row['product_title']),axis=1)

    print("generating feature 2")
    #feature2 : Query exactly occurs in product description (order of words not important)
    train["search_term_in_product_description"]=train.apply(lambda row: exact_similarity(str(row['search_term']),row['product_description']),axis=1)
    test["search_term_in_product_description"]=test.apply(lambda row: exact_similarity(str(row['search_term']),row['product_description']),axis=1)

    print("generating feature 3")
    #feature3: number of common words between search term and product title
    train["number_common_word_product_title"]=train.apply(lambda row: common_words(row['search_term'],row['product_title']),axis=1)
    test["number_common_word_product_title"]=test.apply(lambda row: common_words(row['search_term'],row['product_title']),axis=1)

    print("generating feature 4")
    #feature4: number of common words between search term and product description
    train['number_common_word_product_description']=train.apply(lambda row: common_words(row['search_term'],row['product_description']),axis=1)
    test['number_common_word_product_description']=test.apply(lambda row: common_words(row['search_term'],row['product_description']),axis=1)

    print("generating feature 5")
    #feature5: length of search term
    train['len_search_term']=train.search_term.map(lambda x: 1.0*len(x.split()))
    test['len_search_term']=test.search_term.map(lambda x: 1.0*len(x.split()))

    print("generating feature 6")
    #feature6: length of title
    train['len_product_title']=train.product_title.map(lambda x: 1.0*len(x.split()))
    test['len_product_title']=test.product_title.map(lambda x: 1.0*len(x.split()))

    print("generating feature 7")
    #feature7: length of product_description
    train['len_product_description']=train.product_description.map(lambda x: 1.0*len(x.split()))
    test['len_product_description']=test.product_description.map(lambda x: 1.0*len(x.split()))

    print("generating feature 8")
    #feature8: ratio of common words between search term and product title
    train["ratio_common_word_product_title"]=train["number_common_word_product_title"]/(train['len_product_title']+train['len_search_term']-train["number_common_word_product_title"])
    train.loc[train["ratio_common_word_product_title"].isnull(), "ratio_common_word_product_title"] = -1
    test["ratio_common_word_product_title"]=test["number_common_word_product_title"]/(test['len_product_title']+test['len_search_term']-test["number_common_word_product_title"])
    test.loc[test["ratio_common_word_product_title"].isnull(), "ratio_common_word_product_title"] = -1

    print("generating feature 9")
    #feature9: ratio of common words between search term and product description
    train["ratio_common_word_product_description"]=train["number_common_word_product_description"]/(train['len_product_description']+train['len_search_term']-train["number_common_word_product_description"])
    test["ratio_common_word_product_description"]=test["number_common_word_product_description"]/(test['len_product_description']+test['len_search_term']-test["number_common_word_product_description"])

    data = train.append(test, ignore_index=True)

    print("generating features 10-14")
    #feature: generate term frequency(AKA bag of words)"
    #generate common bag of words
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(data["all_text"].values)

    #feature10-14: summary of term frequency in product_title
    X = vectorizer.transform(train["product_title"].values)
    train['sum_tf_product_title'] = X.sum(axis=1)
    train['max_tf_product_title'] = X.max(axis=1).todense()
    train['min_tf_product_title'] = X.min(axis=1).todense()
    train['mean_tf_product_title'] = X.mean(axis=1)
    train['var_tf_product_title'] = StandardScaler(with_mean=False).fit(X.transpose()).var_
    
    X = vectorizer.transform(test["product_title"].values)
    test['sum_tf_product_title'] = X.sum(axis=1)
    test['max_tf_product_title'] = X.max(axis=1).todense()
    test['min_tf_product_title'] = X.min(axis=1).todense()
    test['mean_tf_product_title'] = X.mean(axis=1)
    test['var_tf_product_title'] = StandardScaler(with_mean=False).fit(X.transpose()).var_

    #feature15-19: summary of term frequency in product_description
    print("generating features 15-19")
    X = vectorizer.transform(train["product_description"].values)
    train['sum_tf_product_description'] = X.sum(axis=1)
    train['max_tf_product_description'] = X.max(axis=1).todense()
    train['min_tf_product_description'] = X.min(axis=1).todense()
    train['mean_tf_product_description'] = X.mean(axis=1)
    train['var_tf_product_description'] = StandardScaler(with_mean=False).fit(X.transpose()).var_
    
    X = vectorizer.transform(test["product_description"].values)
    test['sum_tf_product_description'] = X.sum(axis=1)
    test['max_tf_product_description'] = X.max(axis=1).todense()
    test['min_tf_product_description'] = X.min(axis=1).todense()
    test['mean_tf_product_description'] = X.mean(axis=1)
    test['var_tf_product_description'] = StandardScaler(with_mean=False).fit(X.transpose()).var_

    print("generating features 20-29")
    #feature: genereate term frequency(AKA bag of words)"
    #generate common bag of words
    vectorizer = TfidfVectorizer(ngram_range=(1, 1),stop_words='english')
    vectorizer.fit(data["all_text"].values)
    tsvd = TruncatedSVD(n_components=2, random_state = 1301)
    tsvd.fit(vectorizer.transform(data["product_title"].values))

    X = vectorizer.transform(train["product_title"].values)
    train['sum_tf_product_title_tfidf'] = X.sum(axis=1)
    train['max_tf_product_title_tfidf'] = X.max(axis=1).todense()
    train['min_tf_product_title_tfidf'] = X.min(axis=1).todense()
    train['mean_tf_product_title_tfidf'] = X.mean(axis=1)
    train['var_tf_product_title_tfidf'] = StandardScaler(with_mean=False).fit(X.transpose()).var_
    
    ddd = pd.DataFrame(tsvd.transform(X))
    #ddd.columns=['t0','t1','t2','t3','t4','t5','t6','t7','t8','t9']
    ddd.columns=['t0','t1']
    train = pd.concat([train, ddd], axis=1)
    
    X = vectorizer.transform(test["product_title"].values)
    test['sum_tf_product_title_tfidf'] = X.sum(axis=1)
    test['max_tf_product_title_tfidf'] = X.max(axis=1).todense()
    test['min_tf_product_title_tfidf'] = X.min(axis=1).todense()
    test['mean_tf_product_title_tfidf'] = X.mean(axis=1)
    test['var_tf_product_title_tfidf'] = StandardScaler(with_mean=False).fit(X.transpose()).var_
    
    ddd = pd.DataFrame(tsvd.transform(X))
    #ddd.columns=['t0','t1','t2','t3','t4','t5','t6','t7','t8','t9']
    ddd.columns=['t0','t1']
    test = pd.concat([test, ddd], axis=1)
    
    tsvd = TruncatedSVD(n_components=2, random_state = 1301)
    tsvd.fit(vectorizer.transform(data["product_description"].values))

    X = vectorizer.transform(train["product_description"].values)
    train['sum_tf_product_description_tfidf'] = X.sum(axis=1)
    train['max_tf_product_description_tfidf'] = X.max(axis=1).todense()
    train['min_tf_product_description_tfidf'] = X.min(axis=1).todense()
    train['mean_tf_product_description_tfidf'] = X.mean(axis=1)
    train['var_tf_product_description_tfidf'] = StandardScaler(with_mean=False).fit(X.transpose()).var_

    ddd = pd.DataFrame(tsvd.transform(X))
    #ddd.columns=['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9']
    ddd.columns=['d0','d1']
    train = pd.concat([train, ddd], axis=1)
    
    X = vectorizer.transform(test["product_description"].values)
    test['sum_tf_product_description_tfidf'] = X.sum(axis=1)
    test['max_tf_product_description_tfidf'] = X.max(axis=1).todense()
    test['min_tf_product_description_tfidf'] = X.min(axis=1).todense()
    test['mean_tf_product_description_tfidf'] = X.mean(axis=1)
    test['var_tf_product_description_tfidf'] = StandardScaler(with_mean=False).fit(X.transpose()).var_
    
    ddd = pd.DataFrame(tsvd.transform(X))
    #ddd.columns=['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9']
    ddd.columns=['d0','d1']
    test = pd.concat([test, ddd], axis=1)
    
    print("generating features 30")
    X = vectorizer.transform(train["product_title"])
    Y = vectorizer.transform(train["search_term"])
    Z=np.zeros((train.shape[0],1))
    for i,(x,y) in enumerate(zip(X,Y)):
        Z[i]=cosine_similarity(x,y)
    train["cos_similarity_search_and_title"] = Z
    
    X = vectorizer.transform(test["product_title"])
    Y = vectorizer.transform(test["search_term"])
    Z=np.zeros((test.shape[0],1))
    for i,(x,y) in enumerate(zip(X,Y)):
        Z[i]=cosine_similarity(x,y)
    test["cos_similarity_search_and_title"] = Z

    print("generating features 31")
    X = vectorizer.transform(train["product_description"])
    Y = vectorizer.transform(train["search_term"])
    Z=np.zeros((train.shape[0],1))
    for i,(x,y) in enumerate(zip(X,Y)):
        Z[i]=cosine_similarity(x,y)
    train["cos_similarity_search_and_description"] = Z
    
    X = vectorizer.transform(test["product_description"])
    Y = vectorizer.transform(test["search_term"])
    Z=np.zeros((test.shape[0],1))
    for i,(x,y) in enumerate(zip(X,Y)):
        Z[i]=cosine_similarity(x,y)
    test["cos_similarity_search_and_description"] = Z

    print("generating feature 32")
    df_attributes = pd.read_csv(config.original_attributes_data_path,encoding=config.encoding)
    df_brand = df_attributes[df_attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "product_brand"})
    df_brand['product_brand'] = df_brand['product_brand'].astype(str).str.strip()
    df_brand["product_brand"] = df_brand.apply(lambda row: normalize(row["product_brand"]),axis=1)
    df_brand["product_brand"] = df_brand.apply(lambda row: stem(row["product_brand"]),axis=1)
    df_brand["product_brand"] = df_brand.apply(lambda row: tokens_with_pos(row["product_brand"]),axis=1)
    
    train = pd.merge(train, df_brand, how='left', on='product_uid')
    train.product_brand.fillna('nobrand',inplace = True)
    train['attr'] = train['search_term']+"\t"+train['product_brand']
    train['word_in_brand'] = train['attr'].map(lambda x: similar(x.split('\t')[0],x.split('\t')[1]))
    train['len_of_brand'] = train['product_brand'].map(lambda x:len(x.split())).astype(np.int64)
    train['ratio_brand'] = train['word_in_brand']/train['len_of_brand']
    train.loc[train['ratio_brand'].isnull(), 'ratio_brand'] = -1
    train.loc[train['ratio_brand'].isin([np.inf, -np.inf]), 'ratio_brand'] = -1
    
    test = pd.merge(test, df_brand, how='left', on='product_uid')
    test.product_brand.fillna('nobrand',inplace = True)
    test['attr'] = test['search_term']+"\t"+test['product_brand']
    test['word_in_brand'] = test['attr'].map(lambda x:similar(x.split('\t')[0],x.split('\t')[1]))
    test['len_of_brand'] = test['product_brand'].map(lambda x:len(x.split())).astype(np.int64)
    test['ratio_brand'] = test['word_in_brand']/test['len_of_brand']
    test.loc[test['ratio_brand'].isnull(), 'ratio_brand'] = -1
    test.loc[test['ratio_brand'].isin([np.inf, -np.inf]), 'ratio_brand'] = -1

    #product brand
    le = preprocessing.LabelEncoder()
    product_brand = train[["product_brand"]].append(test[["product_brand"]], ignore_index=True)
    product_brand = product_brand["product_brand"].drop_duplicates()
    le.fit(product_brand)
    train['product_brand_num']=le.transform(train['product_brand'])
    test['product_brand_num']=le.transform(test['product_brand'])

    train['search_term_feature'] = train['search_term'].map(lambda x:len(x))
    test['search_term_feature'] = test['search_term'].map(lambda x:len(x))
    train = train.drop(['id','product_uid','relevance','search_term','product_title','product_description',
                        'all_text','attr','product_brand'],axis=1)
    test = test.drop(['id','product_uid','relevance','search_term','product_title','product_description',
                      'all_text','attr','product_brand'],axis=1)
    print("Done: generate_features_set5")

    return train, test

if __name__ == "__main__":
    print("Loading data...")
    with open(config.output_folder+"/Dataset.pkl","rb") as f:
        data = cPickle.load(f)

    dfTrain = pd.read_csv(config.original_train_data_path,encoding=config.encoding)
    m = len(dfTrain)
    del dfTrain

    train = data.loc[0:(m-1)].copy()
    train = preprocess_data(train, 'train')
    test = data.loc[m:(len(data)-1)].copy()
    test = preprocess_data(test, 'test')

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    train, test = generate_features(train, test)

    print("Saving train data...")
    with open(config.output_folder+"/XFeat5train.pkl","wb") as f:
        cPickle.dump(train,f,-1)

    print("Saving test data...")
    with open(config.output_folder+"/XFeat5test.pkl","wb") as f:
        cPickle.dump(test,f,-1)
    print("Done.")
