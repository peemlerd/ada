import numpy as np
import pandas as pd
pd.options.display.float_format = '{:20,.2f}'.format
import matplotlib.pyplot as plt
import altair as alt
import pickle
import string
import random

# Statistics library
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score

##### Numerical value pre-processing ######
def truncateColumn(df, lower_bound, upper_bound, col_name):
    """
    @param df: A dataframe whose column we want to truncate
    @param lower_bound: The minimum value we want our column to have.
    @param upper_bound: The maximum value we want our column to have.
    @param col_name: string name of the column we want to truncate
    Return Non
    Usage: Use to handle outliers / wrong stuff with readability.
    """
    assert col_name in df.columns.tolist()
    temp = []
    lb = lower_bound
    ub = upper_bound
    dfcopy = df.copy()
    for val in df[col_name].tolist():
        if val > ub:
            temp.append(ub)
        elif val < lb:
            temp.append(lb)
        else: # Within the range
            temp.append(val)
    dfcopy[col_name] = temp
    return dfcopy

# Calculate the sumamry statistics
def summaryStatistics(df, col_to_summarize, fare = ["min", "max", "median", "mean"]):
    """
    @param df: Dataframe
    @param col_to_summarize: A list of column names (string) to summarize.
    @param fare: Summary statistics.
    Usage: Computes a summary statistics for specific columns in the dataframe.
    """
    temp = {}
    for col in col_to_summarize:
        temp[col] = fare
    result = df.agg(temp)
    return result

## Return the dictionary of topicIDs or categoryIds.
def getIdDict(isTopicID = True):
    """
    @param isTopicID: A Boolean denoting whether we want to return the dictionary to
    translate topicID or categoryID.
    Return: A dictionary in human-readable words.
    Usage: A helper function in findDistribution function.
    """
    id_in_words = {}
    if isTopicID:
        f = open('topicId.txt','r')
        for line in f:
            splitted = line.split("\t")
            topicId = splitted[0]
            definition = splitted[1].strip()
            id_in_words[topicId] = definition
    else:
        f = open('categoryId.txt', 'r')
        for line in f:
            lsplit = line.split("\t")[0]
            lsplit = lsplit.split("-")
            categoryId = lsplit[0].strip(" ")
            category = lsplit[1].strip(" ")
            id_in_words[int(categoryId)] = category.strip() # Recast to int type.
    return id_in_words


def getTopic(relevantTopicId, replacee):
    """
    @param relevantTopicId: String denoting the relevantTopicIds in the dataframe.
                            Each topic is concatenated in this string separated by ",".
    @param replacee: String of symbols to eliminate.
    Return: A list of different strings denoting the topicIds of each video.
    Usage: A helper function to use in displayTopicDistribution function.
    """
    try:
        all_topic = relevantTopicId.split(",")
        temp = []
        for topic in all_topic:
            # Remove every symbol in replacee and white space
            topic = topic.translate(str.maketrans('', '', replacee)).strip()
            temp.append(topic)
        temp = set(temp)
        return list(set(temp))
    except:
        return []

def findDistribution(df, isTopicId = True):
    """
    @param df: Dataframe we want to see the distribution of ID on.
    @param isTopicId: A Boolean variable denoting whether we want a distribution
    of topicId or categoryId.
    Usage: Display the distribution of topicId.
    """
    replacee = string.punctuation.replace("/","") # Every punctuation except backslash.
    replacee = replacee.replace("_","")
    Id_dict = {}
    no_topic = 0
    temp = []

    # Define the types of dictionary: relevantTopicIds
    id_in_words = getIdDict(isTopicId)
    if isTopicId:
        col_name = "relevantTopicIds"
    else:
        col_name = "categoryId"

    for i in range(df[col_name].shape[0]): # Iterate through all videos
        if isTopicId: # Create list of topics to iterate through
            all_Id = getTopic(df[col_name].iloc[i], replacee) # List of Ids for that video
        else:
            all_Id = [df[col_name].iloc[i]]

        for Id in all_Id: # Create a dictionary of Id distribution.
            topic = id_in_words[Id]
            try:
                Id_dict[topic] += 1
            except:
                Id_dict[topic] = 1
        # End for
        # Create a list of which topicId a video receives.
        temp.append([id_in_words[Id] for Id in all_Id])
        if len(all_Id) == 0:
            no_topic += 1
    # End for
    # Display immediate results
    print("\nThere are %d videos (%.2f percent) with no %s"
          %(no_topic, no_topic/df[col_name].shape[0] * 100, col_name))
    avg_topicPerVid = np.average(np.array([len(topic) for topic in temp]))
    print("On average, each video has %.2f topics" %(avg_topicPerVid))
    print("\nReturning topic_per_vid and topic distribution dictionary...")
    return temp, Id_dict


def cleanText(text, stopword_lst, return_string = False):
    """
    @param text: A string of text to clean by removing punctuations, stopwords,
    and splitting bad tails.
    @param stopword_lst: A list of words we wish to remove. Usually used the
    list of stopwords in nltk + sth.
    @param return_string: A Boolean indicating we want to return a list of words
    or a long string.
    Return: A list of keywords for each sentence
    NOTE: Use return_string = True for word cloud; False for topic modelling
    (bag of words).
    """
    text_lst = str(text).split("\\n") # Youtube subtitle denotes lines as \\n, so cannot directly remove punctuations.
    temp = []
    # Create a list of words for each video's subtitle, excluding all stopwords.
    for text in text_lst:
        temp += [word for word in simple_preprocess(text, deacc=True) if word not in stopword_lst]
    # For word cloud or topic modelling.
    if return_string:
        return ' '.join(temp)
    return temp

def generateWordCloud(df, col_name):
    """
    @param df: A dataframe whose column consists of texts we want to clean
    @param col_name: A string of column name whose value is a text we want to clean.
    Usage: Generate a word cloud showing the most frequent words appearing in columns of text.
    NOTE: To generate a word cloud for a specific video, simply index by conditions,
    such as video_id, channel creator, understandable, actionable, etc.
    """
    assert col_name in df.columns.tolist()
    # Pre-process the text into long string.
    text = ""
    for video_subtitle in df[col_name].tolist():
        text += cleanText(video_subtitle, stopword_lst, return_string = True)

    # Generate a word cloud
    wordcloud = WordCloud(background_color="white", max_words=5000,
    contour_width=3, contour_color='steelblue')
    wordcloud.generate(text)
    # Visualize a word cloud
    wordcloud.to_image()
    plt.figure(figsize = (9,6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def testClassifier(X, y, model_name, parameters = [], test_prop = 0.2):
    """
    @param X > X_train, X_calib, X_test
    @param y > y_train, y_calib, y_test
    @param model: String of the name of models we want to test
    Usage: This function is used to test the performance of a given classifier on confusion matrix,
    precision, recall, f-score, etc. We allow fine-tuning to happen within this function using gridSearchCV.
    Return: None
    NOTE: We do not allow make_pipeline(StandardScaler(), classifier) for now.
    """
    # Set random_state = 1 to compare between models.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_prop, random_state = 1)

    # Create and cross-validate the models over parameter space.
    # The cross-validated dataset is X_train, y_train.
    print("Below are the results of %s" %(model_name))

    if model_name == "Logistic":
        model = LogisticRegression()
    elif model_name == "SVM":
        model = GridSearchCV(SVC(probability = True),parameters)
    elif model_name == "RandomForest":
        model = GridSearchCV(RandomForestClassifier(), parameters)

    # Start predicting on test set using best model on X_test, y_test.
    model.fit(X_train, y_train)
    try:
        best = model.best_estimator_
    except:
        best = model
    yhat = best.predict(X_test)

    print("Displaying prediction")
    # Display prediction result as follows: confusion matrix, accuracy, precision, recall, fscore
    prediction = list(map(round, yhat))

    ## Confusion matrix
    cm = confusion_matrix(y_test, prediction)
    tn, fp, fn, tp = cm.ravel() # Read from top left > bottom right
    print ("Confusion Matrix : \n", cm)

    ## Accuracy, precision, etc.
    print('Accuracy =  %.2f' %(accuracy_score(y_test, prediction)))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, prediction, pos_label = 1,
                                                                         average = "binary")
    print("Precision = %.2f\n Recall = %.2f\n F-score = %.2f" %(precision, recall, fscore))

    return X_test, y_test, model

def createROC(X_test, y_test, model, name):
    """
    @param X_test: The dataset used for testing
    @param y_test: The labelled dataset
    @param model: Model we want to find performance
    @param name: (Somewhat redundant) Name to display in the ROC curve plot.
    Create the ROC curve for each of the models and also draw X = y.
    """
    # Plotting the ROC curve for logistic regression
    assert name in ["Logistic", "RandomForest", "SVM"]
    prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    print("The AUC value for %s is %f" %(name, auc))
    ax, fig = plt.subplots(figsize=(9, 6))
    fig = plt.plot(lr_fpr, lr_tpr, marker='.', label=name)
    fig = plt.plot(majority, majority, marker = "*", label="Majority")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve for %s" %(name))
    plt.legend()
    plt.savefig("%s_ROC" %(name))

if __name__=="__main__":
   main()
