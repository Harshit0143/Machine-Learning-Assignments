#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import os
import PIL
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

#Twitter_tain_100.csv has a typo in the column heading. I fixed that manually 

def __read_input__(__PATH_TEXT_DATA__): # Hard coded for CoronaTweet
    types = {
        "Sentiment": "str",
        "CoronaTweet": "str"
    }
    return pd.read_csv(__PATH_TEXT_DATA__ , escapechar = "\\" , usecols = ["Sentiment" , "CoronaTweet"]  , dtype = types)

def __read_input_general__(__PATH_TEXT_DATA__): # hard coded for domain adaptation
    types = {
        "Sentiment": "str",
        "Tweet": "str"
    }
    data =  pd.read_csv(__PATH_TEXT_DATA__ , escapechar = "\\" , usecols = ["Sentiment" , "Tweet"]  , dtype = types)
    data.columns = ["Sentiment" , "CoronaTweet"]
    return data

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")


def print_state(message):
    global use_lemmatizer_remove_stop, use_bigrams , use_trigrams
    print(message)
    print("Lemmatization and Stop Word removal:" , use_lemmatizer_remove_stop)
    print("Using Bigrams:", use_bigrams)
    print("Using Trigrams:" , use_trigrams)
   


def __clean_text__(text , URL_PATTERN = r"https?://\S+|www\.\S+"):
    global use_lemmatizer_remove_stop
    text = text.lower()
    text = re.sub(URL_PATTERN, ' ' , text)
    if (use_lemmatizer_remove_stop):
        return __clean_text_advanced__(text)
    
    text = re.sub(r"[^a-z\s]" , ' ' , text)
    return np.array(text.split())




# can report test and validationset accuracoes with and without remiving @
# clearly shows some level of overfitting
# removed usernames. They don't give new information. Just overfit 
# can mention this in report: accuracy on training set reduces on removing @
def __clean_text_advanced__(text):
    text = re.sub(r"@\w+", ' ', text)
    text = re.sub(r"[^a-z\s]" , ' ' , text)
    text = text.split()
    filtered_text = [word for word in text if word not in stop_words]
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    return lemmatized_text


def __count_word_freq__(words_bag):
    unique_words, word_counts = np.unique(words_bag, return_counts = True)
    word_freq_dict = dict(zip(unique_words, word_counts))
    return word_freq_dict

def __build_vocabulary__(words_freq_dict):
    word_freq_sorted = dict(sorted(words_freq_dict.items(), key = lambda item: item[1], reverse = True))
    vocabulary = {word: index for index, (word, _) in enumerate(word_freq_sorted.items())}
    return vocabulary


def __make_freq_maps_classwise__(classwise_words_bag , vocabulary):
    classwise_word_freq = [__count_word_freq__(classwise_words_bag[i]) for i in range(3)]
    for i in range(3):
        classwise_word_freq[i] = {word: classwise_word_freq[i].get(word, 0) for word in vocabulary.keys()}
    return classwise_word_freq   


def __return_sizes_and_concatenate__(text_rows):
    return len(text_rows) , np.concatenate(text_rows)
    


def __make_bigram_array__(word_list):
    return [word_list[i] + '_' + word_list[i + 1] for i in range(len(word_list) - 1)]


def __make_trigram_array__(word_list):
    return [word_list[i] + '_' + word_list[i + 1] + '_' + word_list[i + 2] for i in range(len(word_list) - 2)]


def __predictor_all_positive__(dataframe, message = ""):  
    all_predictions = np.full(len(dataframe) , "Positive")
    actual = np.array(dataframe["Sentiment"]) 
    accuracy = (np.sum(all_predictions == actual) / len(dataframe)) * 100
    print(message)
    print(f"Accuracy by guessing all Positive: {accuracy:.3f}")
    return all_predictions


def __predictor_random_guess__(dataframe , message = ""):
    print(message)
    all_predictions = prediction_key[np.random.randint(0, 3, size = len(dataframe))]
    actual = np.array(dataframe["Sentiment"]) 
    
    accuracy = (np.sum(all_predictions == actual) / len(dataframe)) * 100
    print(f"Accuracy by guessing randomly: {accuracy:.3f}")
    return all_predictions








def __build_save_word_cloud__(word_freq , save_name , folder = "word_clouds"):
    if (use_lemmatizer_remove_stop):
        save_name += "_cleaned"
    word_cloud = WordCloud(
    background_color = "black",
    max_words = 300,
    max_font_size = 50,
    relative_scaling = 1,
    scale = 2,
    )
    
    word_cloud.generate_from_frequencies(word_freq)

    plt.figure(figsize = (10, 5))
    plt.imshow(word_cloud , interpolation = "bilinear")
    plt.axis("off")
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(os.path.join(folder, save_name) , dpi = 100)
    plt.close()

def __generate_confusion_matrix__(actual , prediction , message = ""):
    print(message)
    print("Total samples:" , len(actual))
    print("Prediction ->", "Neutral" , "Positive" , "Negative")
    print("Actual ↓")
  
    for act in range(3):
        print(prediction_key[act] , end = ' ')
        for pred in range(3):
            cnt = np.sum(np.logical_and(actual == prediction_key[act] , prediction == prediction_key[pred]))
            print(cnt , end = ' ')
        print()
    print()
    print()


def __build_class_priors__(classwise_sample_cnt):
    log_phi = np.array(classwise_sample_cnt , dtype = np.float64)
    log_phi /= np.sum(log_phi)
    log_phi = np.log(log_phi)
    return log_phi
def __build_class_word_probabilities__(class_word_freq , vocabulary):
    phi_k_given_class = np.zeros(len(vocabulary), dtype = np.float64)
    for word in vocabulary:
        phi_k_given_class[vocabulary[word]] =  class_word_freq[word] + 1

    phi_k_given_class /= np.sum(phi_k_given_class)
    phi_k_given_class = np.log(phi_k_given_class)
    return phi_k_given_class

def write_freq_descending(word_freq_map  , filename , folder = "frequency"):
    sorted_words = sorted(word_freq_map.items(), key = lambda x: x[1], reverse = True)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder , filename + ".txt")
    with open(filename, "w") as file:
        for word, frequency in sorted_words:
            file.write(f"{word}: {frequency}\n")



def show(my_dict, file_name , folder = "vacabulary"):
    file_name = os.path.join(folder , file_name + ".txt")
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(file_name, 'w') as file:
        for key, value in my_dict.items():
            line = f"{key}: {value}\n"
            file.write(line)

def merege_matrix(parsed_tweet_list):
    return [word for tweet in parsed_tweet_list for word in tweet]


def __set_options__(lemmatize_remove_stop , bigram , trigram):
    global use_lemmatizer_remove_stop , use_bigrams , use_trigrams
    use_lemmatizer_remove_stop = lemmatize_remove_stop
    use_bigrams  = bigram
    use_trigrams = trigram

class n_gram_data:
    def __init__(self , all_text_rows , classwise_text_rows_array, name = ""):
        self.all_words_bag = merege_matrix(all_text_rows)
        self.all_text_word_freq = __count_word_freq__(self.all_words_bag)
        
        if (use_lemmatizer_remove_stop):
            show(self.all_text_word_freq ,  "vocab_lemmatized_" + name)
            write_freq_descending(self.all_text_word_freq , filename = "lem_" + name)

        else:
            show(self.all_text_word_freq , "vocab_non_lemmatized_" + name)
            write_freq_descending(self.all_text_word_freq , filename = "non_lem_" + name)
            


        self.vocabulary = __build_vocabulary__(self.all_text_word_freq)
      
        self.classwise_words_bag = [merege_matrix(classwise_text_rows_array[i]) for i in range(3)]
        self.classwise_word_freq = __make_freq_maps_classwise__(self.classwise_words_bag , self.vocabulary)
        self.log_class_word_probabilities = [__build_class_word_probabilities__(self.classwise_word_freq[i] , self.vocabulary) for i in range (3)]

    def __probability_feature_match_class_k__(self , parsed_tweet , k):
        tot = 0
        for word in parsed_tweet:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tot += self.log_class_word_probabilities[k][idx]
        return tot
    
    def __save_word_cloud_plots__(self , name = ""):
        __build_save_word_cloud__(self.all_text_word_freq , name + "_cloud_all")
        for i in range(3):
            __build_save_word_cloud__(self.classwise_word_freq[i] , name + "_cloud_" + prediction_key[i])
       
def __build_bigram_lists__(all_text_rows, classwise_text_rows_array):
    all_text_rows_bigram = [__make_bigram_array__(row) for row in all_text_rows]
    classwise_text_rows_array_bigram = [[__make_bigram_array__(row) for row in classwise_text_rows_array[i]] for i in range (3)]
    return all_text_rows_bigram , classwise_text_rows_array_bigram

def __build_trigram_lists__(all_text_rows, classwise_text_rows_array):
    all_text_rows_trigram = [__make_trigram_array__(row) for row in all_text_rows]
    classwise_text_rows_array_trigram = [[__make_trigram_array__(row) for row in classwise_text_rows_array[i]] for i in range (3)]
    return all_text_rows_trigram , classwise_text_rows_array_trigram
    
def __group_data__(all_text_rows , train_data):
    neutral = all_text_rows[train_data["Sentiment"] == "Neutral"]
    positive = all_text_rows[train_data["Sentiment"] == "Positive"]
    negative = all_text_rows[train_data["Sentiment"] == "Negative"]
    return [neutral , positive , negative]



class NaiveBayes:
    # [neutral , positive , negative]
    def __predictor_Naive_Bayes__(self , dataframe , message = ""):
        print(message)
        all_predictions = np.array(dataframe["CoronaTweet"].apply(self.__make_prediction__)) 
        actual = np.array(dataframe["Sentiment"]) 
        accuracy = (np.sum(all_predictions == actual) / len(dataframe)) * 100
        print(f"Accuracy using Naive Bayes: {accuracy:.3f}")
        return all_predictions
    
    
    def __give_accuracy_Naive_Bayes__(self , dataframe):

        all_predictions = np.array(dataframe["CoronaTweet"].apply(self.__make_prediction__)) 
        actual = np.array(dataframe["Sentiment"]) 
        accuracy = (np.sum(all_predictions == actual) / len(dataframe)) * 100
        print(f"Accuracy using Naive Bayes: {accuracy:.3f}")
        return accuracy

  

    def __make_prediction__(self , tweet):
        tweet = __clean_text__(tweet)
        bigram_tweet = __make_bigram_array__(tweet)
        trigram_tweet = __make_trigram_array__(tweet)
        score_class_k = np.zeros(3 , dtype = np.float64)
        for i in range(3):
            score_class_k[i] += self.log_phi[i]
            score_class_k[i] += self.unigram.__probability_feature_match_class_k__(tweet , i)
            if (use_bigrams):
                score_class_k[i] += self.bigram.__probability_feature_match_class_k__(bigram_tweet , i)
            if (use_trigrams):
                score_class_k[i] += self.trigram.__probability_feature_match_class_k__(trigram_tweet , i)


        return prediction_key[np.argmax(score_class_k)]



    def train_model(self , train_data  , lemmatize_remove_stop = False , bigram = False , trigram = False):
        __set_options__(lemmatize_remove_stop , bigram , trigram)
        print_state(f"Training. Training data size: {len(train_data)}")


        all_text_rows_unigram =  np.array(train_data["CoronaTweet"].apply(__clean_text__))
        
        
        classwise_text_rows_array_unigram = __group_data__(all_text_rows_unigram , train_data)
        all_text_rows_bigram , classwise_text_rows_array_bigram =  __build_bigram_lists__(all_text_rows_unigram , classwise_text_rows_array_unigram)
        all_text_rows_trigram , classwise_text_rows_array_trigram =  __build_trigram_lists__(all_text_rows_unigram , classwise_text_rows_array_unigram)


        
        classwise_sample_cnt = [len(classwise_text_rows_array_unigram[i]) for i in range (3)]

        self.log_phi = __build_class_priors__(classwise_sample_cnt)
       

        self.unigram = n_gram_data(all_text_rows_unigram , classwise_text_rows_array_unigram , "unigram")
        self.bigram = n_gram_data(all_text_rows_bigram , classwise_text_rows_array_bigram , "bigram")
        self.trigram = n_gram_data(all_text_rows_trigram , classwise_text_rows_array_trigram , "trigram")
        


       
     


        






def update_stop_words():
    global stop_words
    one_letter_words = [chr(97 + i) for i in range(26)]  # Generate lowercase one-letter words (a to z)
    two_letter_words = [chr(97 + i) + chr(97 + j) for i in range(26) for j in range(26)] 
    stop_words.update(one_letter_words + two_letter_words)

def merge_source_train(source_data_df , __TARGET_DATA_PATH__):
    target_data = __read_input_general__(__TARGET_DATA_PATH__)
    train_data = pd.concat([source_data_df, target_data])
    train_data.reset_index(inplace = True , drop = True)
    return train_data

def run_Domain_Adaptation_with_source(SUBSET_SIZES ,__SOURCE_DATA_PATH__ , __TARGET_DATA_SUBSETS__, __VALIDATION_DATA_PATH__):
    print("Training by including Source Domain")
    source_data = __read_input__(__SOURCE_DATA_PATH__)
    test_data = __read_input_general__(__VALIDATION_DATA_PATH__)
    validation_set_accuracies = []



    for i in range (len(__TARGET_DATA_SUBSETS__)):
        print(f"Running with {SUBSET_SIZES[i]}% target data" )
        train_data = merge_source_train(source_data , __TARGET_DATA_SUBSETS__[i])
        spam_classfier = NaiveBayes()
        spam_classfier.train_model(train_data , True , False , False)
        accuracy = spam_classfier.__give_accuracy_Naive_Bayes__(test_data)
        validation_set_accuracies.append(accuracy)
        print()
    return validation_set_accuracies

def run_Domain_Adaptation_without_source(SUBSET_SIZES , __TARGET_DATA_SUBSETS__, __VALIDATION_DATA_PATH__):
    print("Training by excluding Source Domain")
    test_data = __read_input_general__(__VALIDATION_DATA_PATH__)
    validation_set_accuracies = []

    for i in range (len(__TARGET_DATA_SUBSETS__)):
        print(f"Running with {SUBSET_SIZES[i]}% target data" )
        train_data = __read_input_general__(__TARGET_DATA_SUBSETS__[i])
        spam_classfier = NaiveBayes()
        spam_classfier.train_model(train_data , True , False , False)
        accuracy = spam_classfier.__give_accuracy_Naive_Bayes__(test_data)
        validation_set_accuracies.append(accuracy)
        print()
    return validation_set_accuracies
              
     
def make_save_plot(target_size_percentage , val_with_source  , val_without_source  , save_name = "Accuracy_plots"):
    plt.plot(target_size_percentage, val_with_source , marker = 'o', color = "red" , label = "source domain included in training")
    plt.plot(target_size_percentage, val_without_source , marker = 'o', color = "blue" , label = "source domain not included in training")

    plt.xlabel("%age of target data used to train")
    plt.ylabel("Accuracy, in %")
    plt.legend()
    plt.savefig(save_name + ".png")
    plt.close()





def __clean_text_engineered__(text , URL_PATTERN = r"https?://\S+|www\.\S+"):
    text = text.lower()
    text = re.sub(URL_PATTERN, ' ' , text)
    mentions = re.findall(r'@\w+', text)
    hashtags = re.findall(r'#\w+', text)  
    punctuation = re.findall(r'[!?]', text)



    text = re.sub(r"@\w+", ' ', text)
    # text = re.sub(r"#\w+", ' ', text)
    text = re.sub(r"[^a-z\s]" , ' ' , text)
    text = text.split()
    filtered_text = [word for word in text if word not in stop_words]
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    return lemmatized_text , mentions , hashtags , punctuation




class NaiveBayes_engineered:
    # [neutral , positive , negative]
    def __predictor_Naive_Bayes__(self , dataframe , message = ""):
        print(message)
      
        all_predictions = np.array(dataframe["CoronaTweet"].apply(self.__make_prediction__)) 
        actual = np.array(dataframe["Sentiment"]) 
        accuracy = (np.sum(all_predictions == actual) / len(dataframe)) * 100
        print(f"Accuracy using Naive Bayes: {accuracy:.3f}")
        return all_predictions
    
    
    def __give_accuracy_Naive_Bayes__(self , dataframe):
        all_predictions = np.array(dataframe["CoronaTweet"].apply(self.__make_prediction__)) 
        actual = np.array(dataframe["Sentiment"]) 
        accuracy = (np.sum(all_predictions == actual) / len(dataframe)) * 100
        print(f"Accuracy using Naive Bayes: {accuracy:.3f}")
        return accuracy

  

    def __make_prediction__(self , tweet):
    
        tweet , mentions_ , hashtags_ , punctuation_ = __clean_text_engineered__(tweet)
        score_class_k = np.zeros(3 , dtype = np.float64)
       
   
        for i in range(3):
            score_class_k[i] += self.log_phi[i]
            score_class_k[i] +=  self.unigram.__probability_feature_match_class_k__(tweet , i)
            score_class_k[i] +=  1 * self.punctuation.__probability_feature_match_class_k__(punctuation_ , i)
            # score_class_k[i] += self.hashtags.__probability_feature_match_class_k__(hashtags_ , i)
            # score_class_k[i] += self.mentions.__probability_feature_match_class_k__(mentions_ , i)
            


        return prediction_key[np.argmax(score_class_k)]



    def train_model(self , train_data):
        print("Training using engineered features:")
        
        parsed = train_data["CoronaTweet"].apply(__clean_text_engineered__)
    
        all_text_rows_unigram = np.array(parsed.apply(lambda x: x[0]))
        all_mentions_rows =  np.array(parsed.apply(lambda x: x[1]))
        all_hashtags_rows  = np.array(parsed.apply(lambda x: x[2]))
        all_punctuation_rows  = np.array(parsed.apply(lambda x: x[3]))

    
       
        
        classwise_text_rows_array_unigram = __group_data__(all_text_rows_unigram , train_data)
        classwise_mentions_rows =  __group_data__(all_mentions_rows , train_data)
        classwise_hashtags_rows =  __group_data__(all_hashtags_rows , train_data)
        classwise_punctuation_rows =  __group_data__(all_punctuation_rows , train_data)
    

        
        classwise_sample_cnt = [len(classwise_text_rows_array_unigram[i]) for i in range (3)]
        self.log_phi = __build_class_priors__(classwise_sample_cnt)
       

        self.unigram = n_gram_data(all_text_rows_unigram , classwise_text_rows_array_unigram , "unigram")
        self.mentions = n_gram_data(all_mentions_rows , classwise_mentions_rows , "mentions")
        self.hashtags = n_gram_data(all_hashtags_rows, classwise_hashtags_rows , "hashtags")
        self.punctuation =  n_gram_data(all_punctuation_rows, classwise_punctuation_rows, "punctuation")
   



if __name__ == "__main__":
    __COVID_TRAIN_DATA_PATH__ = "./covid_data/Corona_train.csv"
    __COVID_VALIDATION_DATA_PATH__ = "./covid_data/Corona_validation.csv"

    prediction_key = np.array(["Neutral" , "Positive" , "Negative"])
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    use_lemmatizer_remove_stop = False 
    use_bigrams = False
    use_trigrams = False
    update_stop_words()


   
    ##Problem 1#######################################################################################
    ###(a)############################################################################################
    print("__Naive Naive Bayes__________________________________________________________________________")
    obj = NaiveBayes()
    train_data   = __read_input__(__COVID_TRAIN_DATA_PATH__)
    validation_data = __read_input__(__COVID_VALIDATION_DATA_PATH__)
    print("Validation data size:" , len(validation_data))
    obj.train_model(train_data)
    naive_train = obj.__predictor_Naive_Bayes__(train_data , "Testing on Training data")
    naive_val = obj.__predictor_Naive_Bayes__(validation_data , "Testing on Validation data")
    obj.unigram.__save_word_cloud_plots__(name = "train")

    ##########Use just to make the wordCloud#########################################################
    obj = NaiveBayes()
    obj.train_model(validation_data)
    obj.unigram.__save_word_cloud_plots__(name = "validation")
    ###(b)############################################################################################
    print("\n__Dumb Guessing_________________________________________________________________________")
    random_train = __predictor_random_guess__(train_data , "Testing on Training data")
    random_val = __predictor_random_guess__(validation_data , "Testing on Validation data")

    all_pos_train = __predictor_all_positive__(train_data , "Testing on Training data")
    all_pos_val =   __predictor_all_positive__(validation_data  , "Testing on Validation data")
    
    ###(c)############################################################################################
    print("\n__Confusion Matrices____________________________________________________________________")
    actual_train = np.array(train_data["Sentiment"])
    actual_val = np.array(validation_data["Sentiment"])

    __generate_confusion_matrix__(actual_train , random_train , "Random prediction on training set")
    __generate_confusion_matrix__(actual_val , random_val , "Random prediction on validation set")
    __generate_confusion_matrix__(actual_train , all_pos_train ,"Al  positive prediction on training set")
    __generate_confusion_matrix__(actual_val , all_pos_val , "Al  positive prediction on validation set")
    __generate_confusion_matrix__(actual_train , naive_train , "Naive Bayes on training set")
    __generate_confusion_matrix__(actual_val , naive_val , "Naive Bayes on validation set")

  
    ###(d)############################################################################################
    print(f"\n__Model with Lemmaization and Stop words removal_______________________________________")
    obj = NaiveBayes()
    obj.train_model(train_data , lemmatize_remove_stop = True)
    obj.__predictor_Naive_Bayes__(train_data , "Testing on Train Data a")
    obj.__predictor_Naive_Bayes__(validation_data , "Testing on Validation Data")
    obj.unigram.__save_word_cloud_plots__(name = "train")
    ##########Use just to make the wordCloud#########################################################
    obj = NaiveBayes()
    obj.train_model(validation_data , lemmatize_remove_stop = True)
    obj.unigram.__save_word_cloud_plots__(name = "validation")
    ###(e)############################################################################################
    print(f"\n__Model with added Bigram Feature______________________________________________________")
    obj = NaiveBayes()
    obj.train_model(train_data , lemmatize_remove_stop = True , bigram = True)
    obj.__predictor_Naive_Bayes__(train_data , "Testing on Train Data")
    obj.__predictor_Naive_Bayes__(validation_data , "Testing on Validation Data")

    print(f"\n__Model with added Trigram Feature______________________________________________________")
    obj = NaiveBayes()
    obj.train_model(train_data , lemmatize_remove_stop = True , trigram = True)
    obj.__predictor_Naive_Bayes__(train_data , "Testing on Train Data")
    obj.__predictor_Naive_Bayes__(validation_data , "Testing on Validation Data")

    print(f"\n__Model with added Bigram + Trigram Feature______________________________________________________")
    obj = NaiveBayes()
    obj.train_model(train_data , lemmatize_remove_stop = True , bigram = True, trigram = True)
    obj.__predictor_Naive_Bayes__(train_data , "Testing on Train Data")
    obj.__predictor_Naive_Bayes__(validation_data , "Testing on Validation Data")

    
    print(f"\n__Other Self Engineered Features____________________________________________________________")
    obj = NaiveBayes_engineered()
    train_data   = __read_input__(__COVID_TRAIN_DATA_PATH__)
    validation_data = __read_input__(__COVID_VALIDATION_DATA_PATH__)
    print("Validation data size:" , len(validation_data))
    obj.train_model(train_data)
    obj.__predictor_Naive_Bayes__(train_data , "Testing on Training data")
    obj.__predictor_Naive_Bayes__(validation_data , "Testing on Validation data")


    ###(f)############################################################################################
    print("\n\n\n")
    SUBSET_SIZES = [1 , 2, 5 , 10 , 25 , 50 , 100]
    __TWITTER_TRAIN_DATA_PATHS_ = ["./Domain_Adaptation/Twitter_train_" + str(_) + ".csv" for _ in SUBSET_SIZES]
    __TWITTER_VALIDATION_DATA_PATHS_ = "./Domain_Adaptation/Twitter_validation.csv"
    accuracies_with_soruce = run_Domain_Adaptation_with_source(SUBSET_SIZES , __COVID_TRAIN_DATA_PATH__ , __TWITTER_TRAIN_DATA_PATHS_ , __TWITTER_VALIDATION_DATA_PATHS_)
    accuracies_without_source = run_Domain_Adaptation_without_source(SUBSET_SIZES , __TWITTER_TRAIN_DATA_PATHS_ , __TWITTER_VALIDATION_DATA_PATHS_)
    make_save_plot(SUBSET_SIZES , accuracies_with_soruce , accuracies_without_source)


