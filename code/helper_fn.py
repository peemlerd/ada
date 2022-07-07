import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import string
import random
pd.options.display.float_format = '{:20,.2f}'.format

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score

# Text-processing model
import spacy
import seaborn as sns

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
    @param fare: List of summary statistics to compute.
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
    majority = np.arange(0,1,0.05)
    fig = plt.plot(fpr, tpr, marker='.', label=name)
    fig = plt.plot(majority, majority, marker = "*", label="Majority")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve for %s" %(name))
    plt.legend()
    plt.savefig("%s_ROC" %(name))


def comparisonTable(df, dependent, to_summarize, fare = ["min", "max", "median", "mean", "std"] ):
    """
    @param df: A dataframe to compare between two groups
    @param dependent: Dependent variable to create comparison on. Should be "info","action",
                    "understand", "misinfo"
    @param to_summarize: A list of column names to summarize (in this case, all numeric)
    @param fare: What to summarize on.
    Usage: Create summary statistics to compare between two groups of dependent variables.
    Return: A dataframe with Row_i, Row_{i+1} as ARI_info_0, ARI_info_1
    """
    assert dependent in ["info", "action", "understand", "misinformation"]
    # Define df_i as a dataframe with dependent variable = i.
    df_1 = df[df[dependent] == 1]
    df_0 = df[df[dependent] == 0]
    summary_lst = [summaryStatistics(df_0, to_summarize, fare = fare),
                   summaryStatistics(df_1, to_summarize, fare=fare)]
    newdf = pd.DataFrame(index = fare)
    for feature in to_summarize:
        for i in range(2): # Hard code for types of dependent variables.
            col_name = feature + "_%s_%s" %(dependent, str(i))
            newdf[col_name] = summary_lst[i][feature]
    return newdf

def boxplotVisualizer(df, dependent, feature_lst = [], title = "", figsize = (15,10),
                      truncate = False, log_transform = False,
                      isGrid = False, layout = (2,3)):
    """
    @param df: A dataframe whose columns consist of numerical variables we want to visualize (boxplot).
    @param dependent: A string name of dependent variable for which we want to create a boxplot.
    @param feature_lst: A list of feature names (str).
    @param col_category: A string denoting the group of columns we wish to create boxplot altogether,
                         such as readability, viewer engagement. This will appear as suptitle.
    @param log_transform: If we want to log-transform each variable before visualizing.
    ---- Fancy functionality -----
    @param isGrid: A Boolean variable denoting whether we want to stack everything or create one grid.
    @param layout: A tuple og grid dimensions.
    -----------------------------
    Return: A boxplot figure.
    Usage: Identify which numerical variables may be relevant to our classification tasks. For Rema.
    TODO: Implement value truncation
    TODO(Fancy): Grid plot.
    """
    # Checking veracity of input variables.
    assert dependent in ["info", "action", "understand", "misinformation"]
    for col in feature_lst:
        assert col in df.columns.tolist()

    # Create a copy of dataframe to transform variables without messing the dataframe
    dfcopy = df[feature_lst + [dependent]].copy()
    transform_info = ""

    # Apply log-transformation (NOTE: +10 is to prevent error with log(0)).
    if log_transform:
        transform_info = "log_transformed"
        for col in feature_lst:
            dfcopy[col] = np.log10(dfcopy[col] + 10)

    """
    # Generate the indices by which we create plots.
    # NOTE: Plots are created left to right, up to down.
    assert len(feature_lst) == layout[0]*layout[1]
    if isGrid:
        nrow = layout[0]
        ncol = layout[1]
        x_iter = np.repeat(np.arange(0,ncol,1), nrow, axis = 0)
        y_iter = np.tile(np.arange(0,nrow,1), ncol)
    """
    # Create multiple subplots
    fig, ax = plt.subplots(1, len(feature_lst), figsize = figsize)
    fig.suptitle('Boxplots on %s for %s groups (%s).' %(title, dependent.upper(), transform_info))
    for i, col in enumerate(feature_lst):
        """
        if isGrid:
            cur_ax = ax[x_iter[i], y_iter[i]]
        else:
            cur_ax = ax[i]
        """
        # NOTE: If len(feature_lst) = 1, ax returns a subplot itself (hence, cannot index).
        if len(feature_lst) <= 1:
            cur_ax = ax
        else:
            cur_ax = ax[i]
        cur_ax.set_title("%s" %(col))
        cur_ax.grid()
        sns.boxplot(ax = cur_ax, data = dfcopy, x = dependent, y = col)
    return fig

if __name__=="__main__":
   main()
