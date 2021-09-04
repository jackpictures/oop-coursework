
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import PySimpleGUI as sg
import re

layout = [
    [sg.Text('Dataset:'), sg.InputText(), sg.FileBrowse(), sg.Checkbox('DTC'), sg.Checkbox('GNB'), sg.Checkbox('LDA'), sg.Checkbox('QDA'), sg.Checkbox('KNN'), sg.Checkbox('SVC')],
    [sg.Text('\t\t\t\t\t\t\t        '), sg.Checkbox('Logistic Regression')],
    [sg.Text('Plot Graphs:'), sg.Checkbox('Histogram for each feature'), sg.Checkbox('Box plots'),sg.Checkbox('Violin plot'),sg.Checkbox('Pair plot'),sg.Checkbox('DTC plot'),sg.Checkbox('LDA plot'),sg.Checkbox('QDA plot'),sg.Checkbox('Different k for KNN plot')],
    [sg.Output(size=(88, 20))],
    [sg.Submit(), sg.Cancel()]
]
window = sg.Window('Iris classification', layout)
while True:                             
    event, values = window.read()
    if event in (None, 'Exit', 'Cancel'):
        break
    if event == 'Submit':
        file1 =  isitago = None
        print(values[0])
        if values[0]:
            file1 = re.findall('.+:\/.+\.+.', values[0])
            isitago = 1
            if not file1 and file1 is not None:
                print('Error: File path not valid.')
                isitago = 0
            k=0;
            for i in values:
                if values[i]==True : k=1
            if k == 0 : print("Choose smth")
            elif isitago == 1:
                print('Info: Filepath correctly defined.')
                filepaths = [] #files
                filepaths.append(values[0])
                data = pd.read_csv(values[0])
                print(data.head(5))
                print(data.describe())
                print(data.groupby('species').size())


                train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
                X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
                y_train = train.species
                X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
                y_test = test.species
                y_train_en = y_train.replace({'setosa':0,'versicolor':1,'virginica':2}).copy()

                n_bins = 10
                print("\n\n")
                selected_predictors = ["petal_length", "petal_width"]
                algos = [] #algos to compare
                if values[1] == True: algos.append('DTC')
                if values[2] == True: algos.append('GNB')
                if values[3] == True: algos.append('LDA')
                if values[4] == True: algos.append('QDA')
                if values[5] == True: algos.append('KNN')
                if values[6] == True: algos.append('SVC')
                if values[7] == True: algos.append('Logistic Regression')
                if values[8] == True: algos.append('Histogram for each feature')
                if values[9] == True: algos.append('Box plots')
                if values[10] == True: algos.append('Violin plot')
                if values[11] == True: algos.append('Pair plot')
                if values[12] == True: algos.append('DTC plot')
                if values[13] == True: algos.append('LDA plot')
                if values[14] == True: algos.append('QDA plot')
                if values[15] == True: algos.append('Different k for KNN plot')
                fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"] #for plots
                cn = ['setosa', 'versicolor', 'virginica']
                DTC = 0
                LDA = 0
                for algo in algos:
                    if algo == 'DTC':
                        print("DTC Classification:\n")
                        DTC = 1
                        mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
                        mod_dt.fit(X_train,y_train)
                        prediction=mod_dt.predict(X_test)
                        print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
                        print("Importance of each predictor [sepal length,sepal width, petal length, petal width] :")
                        print(mod_dt.feature_importances_)
                        print("\n\n");
                    if algo == 'GNB':
                        # Guassian Naive Bayes Classifier
                        mod_gnb_all = GaussianNB()
                        GNB = 1
                        y_pred = mod_gnb_all.fit(X_train, y_train).predict(X_test)
                        print('The accuracy of the Guassian Naive Bayes Classifier on test data is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
                        
                        print("\n");
                        # Guassian Naive Bayes Classifier with two predictors
                        mod_gnb = GaussianNB()
                        y_pred = mod_gnb.fit(X_train[selected_predictors], y_train).predict(X_test[selected_predictors])
                        print('The accuracy of the Guassian Naive Bayes Classifier with 2 predictors on test data is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
                        print("\n\n");
                    if algo == 'LDA':
                        # LDA Classifier
                        LDA = 1
                        mod_lda_all = LinearDiscriminantAnalysis()
                        y_pred = mod_lda_all.fit(X_train, y_train).predict(X_test)
                        print('The accuracy of the LDA Classifier on test data is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
                        print("\n");
                        # LDA Classifier with two predictors
                        mod_lda = LinearDiscriminantAnalysis()
                        y_pred = mod_lda.fit(X_train[selected_predictors], y_train).predict(X_test[selected_predictors])
                        print('The accuracy of the LDA Classifier with two predictors on test data is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
                        
                        print("\n\n");
                    if algo == 'QDA':
                        # QDA Classifier
                        QDA = 1
                        mod_qda_all = QuadraticDiscriminantAnalysis()
                        y_pred = mod_qda_all.fit(X_train, y_train).predict(X_test)
                        print('The accuracy of the QDA Classifier is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
                        
                        print("\n");

                        # QDA Classifier with two predictors
                        mod_qda = QuadraticDiscriminantAnalysis()
                        y_pred = mod_qda.fit(X_train[selected_predictors], y_train).predict(X_test[selected_predictors])
                        print('The accuracy of the QDA Classifier with two predictors is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
                        
                        print("\n\n");
                    if algo == 'KNN':
                        # KNN, first try 5
                        KNN = 1
                        mod_5nn=KNeighborsClassifier(n_neighbors=5) 
                        mod_5nn.fit(X_train,y_train)
                        prediction=mod_5nn.predict(X_test)
                        print('The accuracy of the 5NN Classifier is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
                        
                        print("\n\n");
                    if algo == 'SVC':
                        # SVC with linear kernel
                        # for SVC, may be impractical beyond tens of thousands of samples
                        SVC1 = 1
                        linear_svc = SVC(kernel='linear').fit(X_train, y_train)
                        prediction=linear_svc.predict(X_test)
                        print('The accuracy of the linear SVC is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
                        print("\n");

                        # SVC with polynomial kernel
                        poly_svc = SVC(kernel='poly', degree = 4).fit(X_train, y_train)
                        prediction=poly_svc.predict(X_test)
                        print('The accuracy of the Poly SVC is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
                        print("\n\n");
                    if algo == 'Logistic Regression':
                        LR = 1
                        mod_lr = LogisticRegression(solver = 'newton-cg').fit(X_train, y_train)
                        prediction=mod_lr.predict(X_test)
                        print('The accuracy of the Logistic Regression is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
                        print("\n\n");
                    if algo == 'Histogram for each feature':
                        fig, axs = plt.subplots(2, 2)
                        axs[0,0].hist(train['sepal_length'], bins = n_bins);
                        axs[0,0].set_title('Sepal Length');
                        axs[0,1].hist(train['sepal_width'], bins = n_bins);
                        axs[0,1].set_title('Sepal Width');
                        axs[1,0].hist(train['petal_length'], bins = n_bins);
                        axs[1,0].set_title('Petal Length');
                        axs[1,1].hist(train['petal_width'], bins = n_bins);
                        axs[1,1].set_title('Petal Width');
                        fig.tight_layout(pad=1.0);
                        
                    if algo == 'Box plots':
                        
                        fig, axs = plt.subplots(2, 2)
                        sns.boxplot(x = 'species', y = 'sepal_length', data = train, order = cn, ax = axs[0,0])
                        sns.boxplot(x = 'species', y = 'sepal_width', data = train, order = cn, ax = axs[0,1])
                        sns.boxplot(x = 'species', y = 'petal_length', data = train, order = cn, ax = axs[1,0])
                        sns.boxplot(x = 'species', y = 'petal_width', data = train,  order = cn, ax = axs[1,1])
                        fig.tight_layout(pad=1.0)
                    if algo == 'Violin plot':
                        plt.figure()
                        sns.violinplot(x="species", y="petal_length", data=train, size=5, order = cn, palette = 'colorblind')
                    if algo == 'Pair plot':
                        sns.pairplot(train, hue="species", height = 2, palette = 'colorblind')
                    if algo == 'DTC plot':
                        if DTC == 0:
                           print("Choose DTC Algorythm for DTC Plot!!!")
                           break
                        plt.figure(figsize = (10,8))
                        plot_tree(mod_dt, feature_names = fn, class_names = cn, filled = True)
                    if algo == 'LDA plot':
                        mod_lda_1 = LinearDiscriminantAnalysis()
                        y_pred = mod_lda_1.fit(X_train[selected_predictors], y_train_en).predict(X_test[selected_predictors])
                        N = 300
                        X = np.linspace(0, 7, N)
                        Y = np.linspace(0, 3, N)
                        X, Y = np.meshgrid(X, Y)
                        g = sns.FacetGrid(test, hue="species", height=5, palette = 'colorblind').map(plt.scatter,"petal_length", "petal_width", ).add_legend()
                        my_ax = g.ax
                        zz = np.array([mod_lda_1.predict(np.array([[xx,yy]])) for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )
                        Z = zz.reshape(X.shape)
                        #Plot the filled and boundary contours
                        my_ax.contourf( X, Y, Z, 2, alpha = .1, colors = ('blue','green','red'))
                        my_ax.contour( X, Y, Z, 2, alpha = 1, colors = ('blue','green','red'))
                        # Add axis and title
                        my_ax.set_xlabel('Petal Length')
                        my_ax.set_ylabel('Petal Width')
                        my_ax.set_title('LDA Decision Boundaries with Test Data');
                    if algo == 'QDA plot':
                        mod_qda_1 = QuadraticDiscriminantAnalysis()
                        y_pred = mod_qda_1.fit(X_train.iloc[:,2:4], y_train_en).predict(X_test.iloc[:,2:4])

                        N = 300
                        X = np.linspace(0, 7, N)
                        Y = np.linspace(0, 3, N)
                        X, Y = np.meshgrid(X, Y)

                        g = sns.FacetGrid(test, hue="species", height=5, palette = 'colorblind').map(plt.scatter,"petal_length", "petal_width", ).add_legend()
                        my_ax = g.ax

                        zz = np.array([mod_qda_1.predict(np.array([[xx,yy]])) for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )
                        Z = zz.reshape(X.shape)

                        #Plot the filled and boundary contours
                        my_ax.contourf( X, Y, Z, 2, alpha = .1, colors = ('blue','green','red'))
                        my_ax.contour( X, Y, Z, 2, alpha = 1, colors = ('blue','green','red'))

                        # Addd axis and title
                        my_ax.set_xlabel('Petal Length')
                        my_ax.set_ylabel('Petal Width')
                        my_ax.set_title('QDA Decision Boundaries with Test Data');


                        plt.figure()
                    if algo == 'Different k for KNN plot':
                        acc_s = pd.Series(dtype = 'float')
                        for i in list(range(1,11)):
                            mod_knn=KNeighborsClassifier(n_neighbors=i) 
                            mod_knn.fit(X_train,y_train)
                            prediction=mod_knn.predict(X_test)
                            acc_s = acc_s.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
    
                        plt.plot(list(range(1,11)), acc_s)
                        plt.suptitle("Test Accuracy vs K")
                        plt.xticks(list(range(1,11)))
                        plt.ylim(0.9,0.98)
                plt.show()
        else:
            print('Please choose dataset.')





# try different k (KNN)

window.close()





