#!/usr/bin/e


# Importing needed libraries

from functools import cache
from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from scipy.spatial.distance import cdist

import streamlit as st

# Part of the page which contains Header of a project with first image as decoration
header = st.container()

# Part of the page in which we read the data from a csv file, List of features and first five elements of a dataaset for user information
dataset = st.container()

# Part of the page which visualizes the data for better understanding
visualization = st.container()

# Part of the page where we change words into numerical representation of them, visualizing and removing outliers and scaling the needed data
removing_outliers = st.container()

# Part of the page where we split the data into training and testing parts and train Logistic regression model
splitting_dataset = st.container()

# Part of the page where we train Random forest classifier model and evaluate this model
random_forest = st.container()

# Part of the page where we train KMeans model
k_means = st.container()

# Part of the page where we do some visualization with KMeans model
clusters_visualization = st.container()

# Part of the page where make conclusions based on visualization of KMeans model
clusters_conclusions = st.container()

# Function that reads the data from a csv file, drops unecessary columns and missing values, checks for duplicate values and returns dataset
# @st.cache()
def get_data(filename):
    data = pd.read_csv(filename)
    data = data.drop(columns=['Unnamed: 0', 'id'], axis=1)

    #Dropping the 310 rows which have null values in the Arrival Delay in Minutes
    data.dropna(axis=0, inplace=True)

    #Making sure that dataset has no duplicates
    data.duplicated().sum()

    return data

# Airline image for decoration
image = Image.open('airline_image.jpg')

# Graph of a elbow_method result, we took an image after running it once, because it takes too much time to finish 
image2 = Image.open('elbow_met.jpg')

with header:
    st.title('What factors lead to customer satisfaction for an Airline?')
    st.image(image, use_column_width=True)
    st.subheader('Can you predict passenger satisfaction?')

with dataset:
    st.header('Airline Passenger Satisfaction dataset consists of following data:')

    st.markdown('* **Customer Type:** loyal customer or disloyal customer\n * **Type of Travel:** personal Travel, Business Travel\n * **Age:** ages of the passengers\n * **Class:** travel class in the plane of the passengers (Business, Eco, Eco Plus)\n * **Flight distance:** the flight distance of this journey\n * **Inflight wifi service:** satisfaction level of the inflight wifi service (0:Not Applicable;1-5)\n * **Departure/Arrival time:** satisfaction level of Departure/Arrival time convenient\n * **Online booking:** satisfaction level of online booking\n * **Gate location:** satisfaction level of Gate location\n * **Food and drink:** satisfaction level of Food and drink\n  * **Online boarding:** satisfaction level of online boarding\n * **Seat comfort:** satisfaction level of Seat comfort\n * **Inflight entertainment:** satisfaction level of inflight entertainment\n * **On-board service:** satisfaction level of On-board service\n * **Leg room service:** satisfaction level of Leg room service\n * **Baggage handling:** satisfaction level of baggage handling\n * **Check-in service:** satisfaction level of Check-in service\n * **Inflight service:** satisfaction level of inflight service\n * **Cleanliness:** satisfaction level of Cleanliness\n * **Departure Delay in Minutes:** minutes delayed when departure\n * **Arrival Delay in Minutes:** minutes delayed when arrival\n * **Satisfaction:** Airline satisfaction level(Satisfaction, neutral or dissatisfaction)')

    # Reading data with the help of a function above
    data = get_data("train.csv")

    st.subheader('Here we can see first five elements of our dataset: ')

    # Visualizing first five elements of a dataset
    st.write(data.head())


with visualization:
    st.subheader('Let\'s visualize our data to understand it better:')

    # Dividing a visualization part of page into two columns with equal widths
    graph1, graph2 = st.columns([1, 1])

    # Visualizing the data
    fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x = "Class", data = data)
    plt.title('Number of customers per class', fontsize=14)
    graph1.pyplot(fig1)

    fig2 = plt.figure(figsize=(10, 5))
    sns.countplot(x = "satisfaction", data = data)
    plt.title('Number of customers by satisfaction level', fontsize=14)
    graph2.pyplot(fig2)

    fig3 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Customer Type',hue="satisfaction",data=data)
    plt.title('Barplot of Satisfaction per Customer Type', fontsize=14)
    graph1.pyplot(fig3)
    
    fig4 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Class',hue="satisfaction",data=data)
    plt.title('Barplot of Satisfaction per Travel Class', fontsize=14)
    graph2.pyplot(fig4)

    
    st.markdown('Just looking to the first graph above we can see that we have about 50k of travelers in business class and about 47k in Eco class, but less than 10k travelers in Eco Plus class.')
    st.markdown('We can see on the second graph about 59k of travelers are neutral or dissatisfied and less than 45k of travelers are satisfied with Airline. If we look at the third graph we can conclude that percentage of neutral or dissatisfied travelers per disloyal customer is a lot bigger than per loyal customer.')
    st.markdown('By looking at the fourth graph we can see that travel class in the plane has a great effect on customer\'s satisfaction level.')


    fig5 = plt.figure(figsize=(10, 5))
    sns.histplot(x='Age',hue="satisfaction",data=data)
    plt.title('Age and satisfaction', fontsize=14)
    st.pyplot(fig5)

    fig6 = plt.figure(figsize=(10, 5))
    sns.histplot(x='Flight Distance',hue="satisfaction",data=data)
    plt.title('Flight Distance and satisfaction', fontsize=14)
    st.pyplot(fig6)

    st.markdown('The older generation is more satisfied with airline. In the second graph we can see the more the Flight Distance the more people satisfied with airline.')


    fig7 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Inflight wifi service',hue="satisfaction",data=data)
    plt.title('Inflight wifi service and satisfaction', fontsize=14)
    st.pyplot(fig7)

    fig8 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Food and drink',hue="satisfaction",data=data)
    plt.title('Food and drink and satisfaction', fontsize=14)
    st.pyplot(fig8)

    fig9 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Seat comfort',hue="satisfaction",data=data)
    plt.title('Seat comfort and satisfaction', fontsize=14)
    st.pyplot(fig9)

    fig10 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Inflight entertainment',hue="satisfaction",data=data)
    plt.title('Inflight entertainment and satisfaction', fontsize=14)
    st.pyplot(fig10)

    fig11 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Cleanliness',hue="satisfaction",data=data)
    plt.title('Cleanliness and satisfaction', fontsize=14)
    st.pyplot(fig11)

    fig12 = plt.figure(figsize=(10, 5))
    sns.countplot(x='On-board service',hue="satisfaction",data=data)
    plt.title('On-board service and satisfaction', fontsize=14)
    st.pyplot(fig12)

    st.markdown('Above we can see some Airline services and satisfaction level of customers. As an example we see that everyone who rated Inflight wifi service as 5 all were satisfied with flight.')

with removing_outliers:

    # Changing words into numbers using LabelEncoder

    labelencoder = LabelEncoder()
    data["Gender"] = labelencoder.fit_transform(data["Gender"])
    data["Customer Type"] = labelencoder.fit_transform(data["Customer Type"])
    data["Type of Travel"] = labelencoder.fit_transform(data["Type of Travel"])
    data["Class"] = labelencoder.fit_transform(data["Class"])
    data["satisfaction"] = labelencoder.fit_transform(data["satisfaction"])
    

    st.subheader('And now we have to deal with outliers! \n We dropped missing values from dataset right after reading the data to a dataframe and made sure we have no duplicated values among the data. ')

    # max values of columns are different so we have to split it so that we can see outliers clearer and then Scale large data using StandardScaler

    # data with max_value = 5 (customer ratings)
    df_1 = data[['Inflight wifi service', 'Departure/Arrival time convenient', "Ease of Online booking", "Gate location", 
           "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", 
           "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]]

    # data with max_value more than 1000
    df_2 = data[['Departure Delay in Minutes', 'Arrival Delay in Minutes',]]

    # data with max_value more than 10000
    df_3 = data[['Flight Distance']]

    # data with max_value approximately 130
    df_4 = data[['Age']]

    # Visualizing outliers for every dataset above

    st.markdown('**Visualizing outliers among customers\' rating for airline services:**')

    outliers_for_services = plt.figure()
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.boxplot(data=df_1, orient="h")
    st.pyplot(outliers_for_services)

    st.markdown('**Visualizing outliers among departure and arrival delay in minutes:**')

    outliers_for_dep_arr_delay = plt.figure()
    sns.set(rc={'figure.figsize':(10.7,7.27)})
    sns.boxplot(data=df_2, orient="h")
    st.pyplot(outliers_for_dep_arr_delay)

    st.markdown('**Visualizing outliers for flight distances:**')

    outliers_for_flight_dist = plt.figure()
    sns.set(rc={'figure.figsize':(10.7,7.27)})
    sns.boxplot(data=df_3, orient="h")
    st.pyplot(outliers_for_flight_dist)

    st.markdown('As we can see we have a lot of outliers that will greatly affect on the furhter work, so we have to deal with them.')

    
    # After that we have to scale our data using StandardScaler
    scaler = StandardScaler()
    data[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = pd.DataFrame(scaler.fit_transform(data[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']]), columns = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'])




    # Changing outliers to NaN values
    for x in data[['Checkin service', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']]:
        q75,q25 = np.percentile(data.loc[:,x],[75,25])
        IQR = q75-q25
    
        max = q75+(3*IQR)
        min = q25-(3*IQR)
    
        data.loc[data[x] < min,x] = np.nan
        data.loc[data[x] > max,x] = np.nan


    # Replacing NaN values to mean value of the column for every corresponding column
    data['Flight Distance'].fillna((data['Flight Distance'].mean()), inplace=True)
    data['Checkin service'].fillna((data['Checkin service'].mean()), inplace=True)
    data['Departure Delay in Minutes'].fillna((data['Departure Delay in Minutes'].mean()), inplace=True)
    data['Arrival Delay in Minutes'].fillna((data['Arrival Delay in Minutes'].mean()), inplace=True)
    data['Age'].fillna((data['Age'].mean()), inplace=True)

    st.markdown('We will find outliers using IQR technique, then change them to NaN values and after we will drop NaN values from dataset: ')

    # Showing code-blocks in a web page
    code1 = '''for x in data[['Checkin service', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']]:
                  q75,q25 = np.percentile(data.loc[:,x],[75,25])
                  IQR = q75-q25
            
                  max = q75+(3*IQR)
                  min = q25-(3*IQR)
            
                  data.loc[data[x] < min,x] = np.nan
                  data.loc[data[x] > max,x] = np.nan '''

    st.code(code1, language='python')

    st.markdown('And just like that we will change NaN values to mean() value of every corresponding column: ')

    code2 = ''' data['Flight Distance'].fillna((data['Flight Distance'].mean()), inplace=True)
                data['Checkin service'].fillna((data['Checkin service'].mean()), inplace=True)
                data['Departure Delay in Minutes'].fillna((data['Departure Delay in Minutes'].mean()), inplace=True)
                data['Arrival Delay in Minutes'].fillna((data['Arrival Delay in Minutes'].mean()), inplace=True)
                data['Age'].fillna((data['Age'].mean()), inplace=True) '''

    st.code(code2, language='python')


with splitting_dataset:
    st.subheader('Here we will split data and create Logistic Regression model:')

    # Dividing data into dependent and independent variables

    # Everything except satisfaction column is independent
    x_1 = data.iloc[:, 0:22].values

    # satisfaction column is dependent, we will predict it relying on the independent variables
    y_1 = data.iloc[:, 22].values

    # Storing data into dataframes
    X = pd.DataFrame(x_1) 
    Y = pd.DataFrame(y_1)

    # Dividing data into training and testing parts (20% - testing data size)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

    # Showing code-blocks in a web page
    code3 = ''' # Spliiting data into dependent and independent parts:\n\n# Independent\nX = pd.DataFrame(data.iloc[:, 0:22].values)\n\n# Dependent\nY = pd.DataFrame(data.iloc[:,22].values)\n\n# Splitting data into training and testing parts\nX_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)\n\n# Creating a model:\nlog_model = LogisticRegression()\nlog_model.fit(X_train, Y_train) '''
    st.code(code3, language='python')

    st.subheader('Evaluating Logistic Regression model:')

    # Dividing splitting_dataset part of the page into two columns 
    # select-col contains model evaluation, display_col contains learning curve of a model and confusion matrix
    select_col, display_col = st.columns(2)

    # Creating LogisticRegression model
    log_model = LogisticRegression()

    # Fitting LogisticRegression model with training part of a data
    log_model.fit(X_train, Y_train)

    # Model evaluation: we do predictions with LogisticRegression for every value based on the X_test, then comparing result of pedictions to our actual Y_test
    select_col.text('Prediction Accuracy:')
    select_col.write(accuracy_score(Y_test, log_model.predict(X_test)))

    select_col.text('Precision:')
    select_col.write(precision_score(Y_test, log_model.predict(X_test)))

    select_col.text('Recall:')
    select_col.write(recall_score(Y_test, log_model.predict(X_test)))

    select_col.text('F1 Score:')
    select_col.write(f1_score(Y_test, log_model.predict(X_test)))

    select_col.text('Mean absolute error:')
    select_col.write(mean_absolute_error(Y_test, log_model.predict(X_test)))

    select_col.text('Mean squared error:')
    select_col.write(mean_squared_error(Y_test, log_model.predict(X_test)))

    select_col.text('R squared score:')
    select_col.write(r2_score(Y_test, log_model.predict(X_test)))

    # Drawing learning curves take a lot of time, we don't want browser to rebuild it every time, so we do caching using @cache before a function
    # Function for learning curve for LogisticRegression
    @cache
    def draw_graph1():
        train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), x_1, y_1, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        fig_curve = plt.figure(figsize=(10, 5))
        plt.plot(train_sizes, train_mean)
        plt.plot(train_sizes, test_mean)

        return fig_curve

    # function for drawing confusion matrix
    @cache
    def conf_matrix():
        return confusion_matrix(Y_test, log_model.predict(X_test))
    
    # Learning curve for LogisticRegression
    display_col.markdown('**Learning curve:**')
    display_col.text("blue line - training accuracy\norange line- testing accuracy")
    display_col.pyplot(draw_graph1())

    # Drawing of confusion matrix
    display_col.markdown('**Confusion matrix:**')
    display_col.write(conf_matrix())


with random_forest:
    st.subheader('Creating Random Forest classification model:')

    code4 = '''# Creating and training a model:\nr_forest = RandomForestClassifier(n_estimators=20)\n\nr_forest.fit(X_train, Y_train) '''
    st.code(code4, language='python')

    st.subheader('Evaluating RandomForestClassifier model:')

    # Dividing random_forest part of the page into two columns
    col1, col2 = st.columns(2)

    # Creating and training a model
    r_forest = RandomForestClassifier(n_estimators=20)
    r_forest.fit(X_train, Y_train)

    # Model evaluation: we do predictions with RandomForestClassifier for every value based on the X_test, then comparing result of pedictions to our actual Y_test
    col1.text('Prediction Accuracy:')
    col1.write(accuracy_score(Y_test, r_forest.predict(X_test)))

    col1.text('Precision:')
    col1.write(precision_score(Y_test, r_forest.predict(X_test)))

    col1.text('Recall:')
    col1.write(recall_score(Y_test, r_forest.predict(X_test)))

    col1.text('F1 Score:')
    col1.write(f1_score(Y_test, r_forest.predict(X_test)))

    col1.text('Mean absolute error:')
    col1.write(mean_absolute_error(Y_test, r_forest.predict(X_test)))

    col1.text('Mean squared error:')
    col1.write(mean_squared_error(Y_test, r_forest.predict(X_test)))

    col1.text('R squared score:')
    col1.write(r2_score(Y_test, r_forest.predict(X_test)))

    @cache
    def draw_graph2():
        train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=20), x_1, y_1, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        fig_curve2 = plt.figure(figsize=(10, 5))
        plt.plot(train_sizes, train_mean)
        plt.plot(train_sizes, test_mean)

        return fig_curve2

    @cache
    def conf_matrix2():
        return confusion_matrix(Y_test, r_forest.predict(X_test))

    # Learning curve for RandomForestClassifier
    col2.markdown('**Learning curve:**')
    col2.text("blue line - training accuracy\norange line- testing accuracy")
    col2.pyplot(draw_graph2())

    # Drawing of confusion matrix for r_forest
    fig_cm = plt.figure(figsize=(10, 5))
    col2.markdown('**Confusion matrix:**')
    col2.write(conf_matrix2())


with k_means:

    st.subheader('Creating KMeans model:')
    st.markdown('Using elbow method we calculated the optimal number of clusters is 8, but we want to choose k = 3')

    # Image is a result of a function below, we decided to make a picture of it because of too long calculations
    st.image(image2, use_column_width=True)

    # Function for finding optimal number of clusters
    @cache
    def elbow_met(data):
        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k, random_state=10).fit(data)
            kmeanModel.fit(data)
            distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

        # Plot the elbow
        clusters_curve = plt.figure(figsize=(10, 5))
        plt.plot(K, distortions, 'bx-')
        st.pyplot(clusters_curve)
        
    
    # Creating KMeans model with 3 Number of clusters = 3
    kmeans = KMeans(n_clusters=3)

    # Fit our data and predict to which claster every datapoint belongs, and then create separate column ["cluster"] in our dataset and store results there
    data["cluster"] = kmeans.fit_predict(data[data.columns[2:]])

    # Showing code-block on a web-page
    code5 = '''# Creating and training a model with k=3 clusters:\nkmeans = KMeans(n_clusters=3)\n# Then predicting cluster for each datapoint in our datase\n# and storing it in a separate column\n data["cluster"] = kmeans.fit_predict(data[data.columns[2:]]) '''
    st.code(code5, language='python')

with clusters_visualization:

    st.subheader('Visualizing clusters:')

    # Dividing clusters_visualization part of a page into two columns, so we could visualize graphs in two columns
    col_1, col_2 = st.columns(2)

    # Visualizing clusters in two columns by every Airline service's rate and see if we can find how data was divided into clusters
    cl_fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Inflight wifi service',hue="cluster",data=data)
    plt.title('Rates of Inflight wifi service per cluster', fontsize=14)
    col_1.pyplot(cl_fig1)

    cl_fig2 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Departure/Arrival time convenient',hue="cluster",data=data)
    plt.title('Rates of departure/Arrival time convenient per cluster', fontsize=14)
    col_2.pyplot(cl_fig2)

    cl_fig3 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Ease of Online booking',hue="cluster",data=data)
    plt.title('Rates of ease of Online booking per cluster', fontsize=14)
    col_1.pyplot(cl_fig3)

    cl_fig4 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Food and drink',hue="cluster",data=data)
    plt.title('Rates of Food and drink per cluster', fontsize=14)
    col_2.pyplot(cl_fig4)

    cl_fig5 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Online boarding',hue="cluster",data=data)
    plt.title('Rates of Online boarding per cluster', fontsize=14)
    col_1.pyplot(cl_fig5)

    cl_fig6 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Seat comfort',hue="cluster",data=data)
    plt.title('Rates of Seat comfort per cluster', fontsize=14)
    col_2.pyplot(cl_fig6)

    cl_fig7 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Inflight entertainment',hue="cluster",data=data)
    plt.title('Rates of Inflight entertainment per cluster', fontsize=14)
    col_1.pyplot(cl_fig7)

    cl_fig8 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Leg room service',hue="cluster",data=data)
    plt.title('Rates of Leg room service per cluster', fontsize=14)
    col_2.pyplot(cl_fig8)

    cl_fig9 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Baggage handling',hue="cluster",data=data)
    plt.title('Rates of Baggage handling per cluster', fontsize=14)
    col_1.pyplot(cl_fig9)

    cl_fig10 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Cleanliness',hue="cluster",data=data)
    plt.title('Rates of Cleanliness per cluster', fontsize=14)
    col_2.pyplot(cl_fig10)

    cl_fig11 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Checkin service',hue="cluster",data=data)
    plt.title('Rates of Check-in service per cluster', fontsize=14)
    col_1.pyplot(cl_fig11)

    cl_fig12 = plt.figure(figsize=(10, 5))
    sns.countplot(x='Inflight service',hue="cluster",data=data)
    plt.title('Rates of Inflight service per cluster', fontsize=14)
    col_2.pyplot(cl_fig12)

with clusters_conclusions:

    st.subheader('Conclusions:')

    # Dividing clusters_conclusions part of a page into three columns to write a description for every cluster based on graphs above
    cluster0, cluster1, cluster2 =  st.columns(3)

    # Cluster 0 description
    cluster0.markdown('**Cluster 0:**')
    cluster0.markdown('* Quite high satisfaction level')
    cluster0.markdown('* High satisfaction in: **Food and Drink**, **Seat comfort**, **Inflight entertainment**, **Cleanliness**, **Inflight service**')
    cluster0.markdown('* Low satisfaction in: **Inflight Wifi service**, **Departure/Arrivale time**, **Ease of online booking**')

    # Cluster 1 description
    cluster1.markdown('**Cluster 1:**')
    cluster1.markdown('* Satisfaction level is slightly below average')
    cluster1.markdown('* Quite high satisfaction in: **Departure/Arrivale time**')
    cluster1.markdown('* Low satisfaction in: **Food and Drink**, **Seat comfort**, **Flight entertainment**, **Cleanliness**')

    # Cluster 2 description
    cluster2.markdown('**Cluster 2:**')
    cluster2.markdown('* Satisfaction level is very high')
    cluster2.markdown('* Quite high satisfaction in: every service was rated high')
    cluster2.markdown('* Most low satisfaction level among other services, but also quite higher than average in: **Checkin services**')



    

    

