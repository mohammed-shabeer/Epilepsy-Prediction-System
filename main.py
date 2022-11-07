from datetime import time
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd
import os
import warnings
import seaborn as sns
import imblearn
from tkinter import *
from tkinter.ttk import Style
import pandas as pd
import pickle
import scipy
import matplotlib.pyplot as plt
import numpy as np
import timeit
'import RPi.GPIO as GPIO'

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer, classification_report
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


warnings.filterwarnings('ignore')


# colours for printing outputs
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

'''GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
ledgreen = 23
ledred = 24
GPIO.setup(ledgreen, GPIO.OUT)
GPIO.setup(ledred, GPIO.OUT)
GPIO.output(ledgreen, GPIO.LOW)
GPIO.output(ledred, GPIO.LOW)'''

# Create the root window
# with specified size and title
root = Tk()
root.title("Epilepsy Prediction System")
root.resizable(0, 0)
canvas1 = Canvas(root, width=400, height=300,  relief='raised')
canvas1.pack()
# root.attributes("-topmost", True) To keep the root window on the front

Output1 = Text(root, height = 2,
              width = 35,
              bg = "lightgrey")
Output1.pack()

label1 = Label(root, text='Epilepsy Prediction System',
               bg='lightgrey', fg='black', font=('Proxima Nova', 12, 'bold'))
canvas1.create_window(200, 25, window=label1)
canvas1.configure(bg='lightgrey')

modelname='None'
# define a function for 2nd toplevel
# window which is not associated with
# any parent window


def tmAda():

    # Read our data (from Dataset)
    data = pd.read_csv(
        'Dataset/Epileptic Seizure Recognition.csv')

    # Change the y target column (make a binary classification task)
    dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
    data['y'] = data['y'].map(dic)

    # Remove Unammed
    data = data.drop('Unnamed', axis=1)

    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy='minority')

    # fit and apply the transform
    X, y = oversample.fit_resample(data.drop('y', axis=1), data['y'])

    # Let us split our dataset on train and test and than invoke validation approach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.3, random_state=42)

    # Initialize Parameters for Classifier
    TEST_SIZE = 0.1
    RANDOM_STATE = 0

    print('ADA Boosting Model Training')
    Output1.delete("1.0","end")
    Output.delete("1.0","end")
    Output1.insert(END, ' Model: ADA Boosting')
    Output.insert(END, 'Model: ADA Boosting')
    

    # Ada Ensemble
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=1,
                                  random_state=RANDOM_STATE)

    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=100,
                             learning_rate=0.1,
                             random_state=RANDOM_STATE)

    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_val)
    print('Training Done')
    acc_bag = round(ada.score(X_train, y_train) * 100, 2)
    acc_text = 'Accuracy is: ' + (str(acc_bag)+' %')
    # Dsiplay Accurary
    #acc_text=display(pd.DataFrame(classification_report(y_train, y_pred , output_dict =True)))

    # Pickle the trained model
    pickle.dump(
        ada, open('Classifier/ADA Boosting Model.pkl', 'wb'))

    global modelname
    modelname='Classifier/ADA Boosting Model.pkl'

    # Window
    showinfo(
        title='Training Done',
        message=acc_text
    )

    # Creating the CSV Selector
    canvas1.create_window(200, 180, window=open_button)


def tmMV():

    # Read our data (from Dataset)
    data = pd.read_csv(
        'Dataset/Epileptic Seizure Recognition.csv')

    # Change the y target column (make a binary classification task)
    dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
    data['y'] = data['y'].map(dic)

    # Remove Unammed
    data = data.drop('Unnamed', axis=1)

    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy='minority')

    # fit and apply the transform
    X, y = oversample.fit_resample(data.drop('y', axis=1), data['y'])

    # Let us split our dataset on train and test and than invoke validation approach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.3, random_state=42)

    # Initialize Parameters for Classifier
    TEST_SIZE = 0.1
    RANDOM_STATE = 0

    print('Majority Voting Model Training')
    Output1.delete("1.0","end")
    Output.delete("1.0","end")
    Output1.insert(END, ' Model: Majority Voting')
    Output.insert(END, 'Model: Majority Voting')
    

    # Majority Voting Ensemble
    clf1 = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(kernel='rbf',
                                 gamma='auto',
                                 random_state=RANDOM_STATE,
                                 probability=True))])

    clf2 = Pipeline([('scl', StandardScaler()),
                     ('clf', LogisticRegression(solver='liblinear',
                                                random_state=RANDOM_STATE))
                     ])

    clf3 = DecisionTreeClassifier(random_state=RANDOM_STATE)

    clf_labels = ['SVM',  # Support Vector Machine
                  'LR',  # LogisticRegression
                  'DT']  # Decision Tree

    # Majority Rule Voting
    hard_mv_clf = VotingClassifier(estimators=[(clf_labels[0], clf1),
                                               (clf_labels[1], clf2),
                                               (clf_labels[2], clf3)],
                                   voting='hard')

    soft_mv_clf = VotingClassifier(estimators=[(clf_labels[0], clf1),
                                               (clf_labels[1], clf2),
                                               (clf_labels[2], clf3)],
                                   voting='soft')

    clf_labels += ['Hard Majority Voting', 'Soft Majority Voting']
    all_clf = [clf1, clf2, clf3, hard_mv_clf, soft_mv_clf]

    print(color.BOLD+color.UNDERLINE+'Validation Scores\n'+color.END)
    for clf, label in zip(all_clf, clf_labels):
        start = timeit.default_timer()  # TIME STUFF

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        scores = f1_score(y_val, y_pred)
        print(color.BOLD+label+color.END)
        print("Score: %0.3f"
              % scores)
        # TIME STUFF
        stop = timeit.default_timer()
        print("Run time:", np.round((stop-start)/60, 2), "minutes")
        print()

    print('Training Done')
    acc_bag = round(clf.score(X_train, y_train) * 100, 2)
    acc_text = 'Accuracy is: ' + (str(acc_bag)+' %')
    # Dsiplay Accurary
    #acc_text=display(pd.DataFrame(classification_report(y_train, y_pred , output_dict =True)))

    # Pickle the trained model
    pickle.dump(
        clf, open('Classifier/Majority Voting Model.pkl', 'wb'))

    global modelname
    modelname='Classifier/Majority Voting Model.pkl'

    # Window
    showinfo(
        title='Training Done',
        message=acc_text
    )

    # Creating the CSV Selector
    canvas1.create_window(200, 180, window=open_button)


def tmBag():

    # Read our data (from Dataset)
    data = pd.read_csv(
        'Dataset/Epileptic Seizure Recognition.csv')

    # Change the y target column (make a binary classification task)
    dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
    data['y'] = data['y'].map(dic)

    # Remove Unammed
    data = data.drop('Unnamed', axis=1)

    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy='minority')

    # fit and apply the transform
    X, y = oversample.fit_resample(data.drop('y', axis=1), data['y'])

    # Let us split our dataset on train and test and than invoke validation approach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.3, random_state=42)

    # Initialize Parameters for Classifier
    TEST_SIZE = 0.1
    RANDOM_STATE = 0

    # Pipeline for Base Estimaters
    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=0.8, random_state=RANDOM_STATE)),
                         ('clf', SVC(kernel='rbf', random_state=RANDOM_STATE))])

    print('Bagging Model Training')
    Output1.delete("1.0","end")
    Output.delete("1.0","end")
    Output1.insert(END, 'Model: Bagging')
    Output.insert(END, 'Model: Bagging')
    # Bagging Ensemble
    bag = BaggingClassifier(base_estimator=pipe_svc,
                            n_estimators=10,
                            max_samples=0.5,
                            max_features=0.5,
                            bootstrap=True,
                            bootstrap_features=True,
                            oob_score=True,
                            warm_start=False,
                            n_jobs=-1,
                            random_state=RANDOM_STATE)
    bag.fit(X_train, y_train)

    # Predict Value
    y_pred = bag.predict(X_train)
    print('Training Done')
    
    acc_bag = round(bag.score(X_train, y_train) * 100, 2)
    acc_text = 'Accuracy : ' + (str(acc_bag)+' %')
    # Dsiplay Accurary
    #acc_text=display(pd.DataFrame(classification_report(y_train, y_pred , output_dict =True)))

    # Pickle the trained model
    pickle.dump(
        bag, open('Classifier/Bagging Model.pkl', 'wb'))

    global modelname
    modelname='Classifier/Bagging Model.pkl'

    # Window
    showinfo(
        title='Training Done',
        message=acc_text
    )

    # Creating the CSV Selector
    canvas1.create_window(200, 180, window=open_button)


def tmRF():

    # Read our data (from Dataset)
    data = pd.read_csv(
        'Dataset/Epileptic Seizure Recognition.csv')

    # Change the y target column (make a binary classification task)
    dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
    data['y'] = data['y'].map(dic)

    # Remove Unammed
    data = data.drop('Unnamed', axis=1)

    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy='minority')

    # fit and apply the transform
    X, y = oversample.fit_resample(data.drop('y', axis=1), data['y'])

    # Let us split our dataset on train and test and than invoke validation approach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.3, random_state=42)

    # Initialize Parameters for Classifier
    TEST_SIZE = 0.1
    RANDOM_STATE = 0
    print('Random Forest Model Training')
    Output1.delete("1.0","end")
    Output.delete("1.0","end")
    Output1.insert(END, 'Model: Random Forest')
    Output.insert(END, 'Model: Random Forest')
    

    # Random Forest Ensemble
    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=1000,
                                    max_features='sqrt',
                                    class_weight='balanced',
                                    random_state=RANDOM_STATE,
                                    n_jobs=-1)
    forest.fit(X_train, y_train)

    y_pred = forest.predict(X_val)

    # Predict Value
    #y_pred = forest.predict(X_train)
    print('Training Done')
    acc_bag = round(forest.score(X_train, y_train) * 100, 2)
    acc_text = 'Accuracy is: ' + (str(acc_bag)+' %')
    # Dsiplay Accurary
    #acc_text=display(pd.DataFrame(classification_report(y_train, y_pred , output_dict =True)))

    # Pickle the trained model
    pickle.dump(forest, open(
        'Classifier/Random Forest Model.pkl', 'wb'))

    global modelname
    modelname='Classifier/Random Forest Model.pkl'

    # Window
    showinfo(
        title='Training Done',
        message=acc_text
    )

    # Creating the CSV Selector
    canvas1.create_window(200, 180, window=open_button)

def tmEnsemble():

    # Read our data (from Dataset)
    data = pd.read_csv(
        'Dataset/Epileptic Seizure Recognition.csv')

    # Change the y target column (make a binary classification task)
    dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
    data['y'] = data['y'].map(dic)

    # Remove Unammed
    data = data.drop('Unnamed', axis=1)

    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy='minority')

    # fit and apply the transform
    X, y = oversample.fit_resample(data.drop('y', axis=1), data['y'])

    # Let us split our dataset on train and test and than invoke validation approach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.3, random_state=42)

    # Initialize Parameters for Classifier
    TEST_SIZE = 0.1
    RANDOM_STATE = 0

    print('Majority Voting Training using Pre-Trained Models')
    Output1.delete("1.0","end")
    Output.delete("1.0","end")
    Output1.insert(END, 'Model: Majority Voting[ADA,BAG,RF]')
    Output.insert(END, 'Model: Majority Voting[ADA,BAG,RF]')
    
    #Loading Model
    ada= pickle.load(open('Classifier/ADA Boosting Model.pkl', "rb"))
    bag= pickle.load(open('Classifier/Bagging Model.pkl', "rb"))
    rft= pickle.load(open('Classifier/Random Forest Model.pkl', "rb"))

    
    # Initialising Models
    models = list()
    models.append(('ada', ada))
    models.append(('bag', bag))
    models.append(('rft', rft))
    
    # define the hard voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='hard')
    
    # fit the model on all available data
    ensemble.fit(X_train, y_train)

    print('Training Done')
    acc_ensemble = round(ensemble.score(X_train, y_train) * 100, 2)
    acc_text = 'Accuracy is: ' + (str(acc_ensemble)+' %')
    
    # Pickle the trained model
    pickle.dump(
        ensemble, open('Classifier/Ensemble Model.pkl', 'wb'))

    global modelname
    modelname='Classifier/Ensemble Model.pkl'

    # Window
    showinfo(
        title='Training Done',
        message=acc_text
    )

    # Creating the CSV Selector
    canvas1.create_window(200, 180, window=open_button)

def predictSeizure():
    # Selecting the File
    filetypes = (
        ('CSV Files', '*.csv'),
        ('All files', '*.*')
    )
    filename = fd.askopenfilename(
        title='Select Patient Record to Predict',
        initialdir='Input/',
        filetypes=filetypes)

    # read the new input from csv file
    new_input = pd.read_csv(filename)
    
    #Plotting
    X = new_input.values
    plt.figure(figsize=(8, 4))
    plt.plot(X[0, :], label='EEG Data')
    plt.legend()
    plt.show()
    
    # load the trained model
    model = pickle.load(
        open(modelname, 'rb'))
    print(modelname)
    # Predict the data
    new_output = model.predict(new_input)

    
    print("LED turning off.")
    '''GPIO.output(ledgreen, GPIO.LOW)
    GPIO.output(ledred, GPIO.LOW) '''
     
    # Printing the Result
    if new_output == [1]:
        print('You Might get Seizure, Be conscious about it')
        '''GPIO.output(ledgreen, GPIO.HIGH)
        GPIO.output(ledred, GPIO.LOW)'''
        # Window
        showinfo(
            title='Test Result',
            message='You Might get Seizure, Be conscious about it.'
        )
        '''GPIO.output(ledgreen, GPIO.LOW)
        GPIO.output(ledred, GPIO.LOW)'''
    else:
        print('You are Safe. No Worries')
        '''GPIO.output(ledgreen, GPIO.LOW)
        GPIO.output(ledred, GPIO.HIGH)'''
        # Window
        showinfo(
            title='Test Result',
            message='You are Safe. No Worries'
        )
        '''GPIO.output(ledgreen, GPIO.LOW)
        GPIO.output(ledred, GPIO.LOW)'''


def chooseModel():
    filetypes = (
        ('CSV Files', '*.pkl'),
        ('All files', '*.*')
    )
    global modelname
    modelname = fd.askopenfilename(
        title='Select Pre-Trained Model',
        initialdir='Classifier/',
        filetypes=filetypes)
    canvas1.create_window(200, 180, window=open_button)
    print(modelname)
    modelnamepath=os.getcwd()
    print(modelnamepath)
    if(modelname==modelnamepath+'/'+'Classifier/ADA Boosting Model.pkl'):
        Output1.delete("1.0","end")
        Output1.insert(END, 'Model: ADA Boosting')
    if(modelname==modelnamepath+'/''Classifier/Bagging Model.pkl'):
        Output1.delete("1.0","end")
        Output1.insert(END, 'Model: Bagging')
    if(modelname==modelnamepath+'/''Classifier/Majority Voting Model.pkl'):
        Output1.delete("1.0","end")
        Output1.insert(END, 'Model: Majority Voting')
    if(modelname==modelnamepath+'/''Classifier/Random Forest Model.pkl'):
        Output1.delete("1.0","end")
        Output1.insert(END, 'Model: Random Forest')
    if(modelname==modelnamepath+'/''Classifier/Ensemble Model.pkl'):
        Output1.delete("1.0","end")
        Output1.insert(END, 'Model: Majority Voting (ADA,BAG,RF)')
#  define a function for 1st toplevel
# which is associated with root window.


def trainModel():

    # Create widget
    top1 = Toplevel(root)
    global Output
    Output = Text(top1, height = 2,
              width = 35,
              bg = "lightgrey")
    Output.pack()
    # Define title for window
    top1.title("Ensemble Method Selection")
    # specify size
    top1.geometry("620x460")
    top1.resizable(0, 0)
    top1.configure(bg='lightgrey')
    # Create label
    label = Label(top1,
                  text="Select Ensemble Method:", bg='lightgrey', font=('Proxima Nova', 17, 'bold'))

    # Create Exit button
    button1 = Button(top1, text="Exit",  bg='black', fg='white', font=('Proxima Nova', 12, 'bold'),
                     command=top1.destroy)
    
    # create button to open toplevel2
    button2 = Button(top1, text="  Random Forest  ",   bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'),
                     command=tmRF)
    button2.place(x=100, y=100)
    button3 = Button(top1, text="Majority Voting\n [SVM, LR, DT]",  bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'),
                     command=tmMV)

    button4 = Button(top1, text="Bagging Classifer", bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'),
                     command=tmBag)

    button5 = Button(top1, text="   ADA Boosting   ",  bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'),
                     command=tmAda)
    button6 = Button(top1, text="Majority Voting\n [ADA, BAG. RF]",  bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'),
                     command=tmEnsemble)

    label.pack(side=TOP)
    
    button6.pack()
    button6.place(x=400, y=280)
    
    button5.pack()
    button5.place(x=35, y=120)# x=left right, y= up down
    #, ipadx=18, ipady=1)

    button4.pack()
    button4.place(x=35, y=240,)
    #side=LEFT,padx=15, pady=15, ipadx=5, ipady=1)
    
    button3.pack()
    button3.place(x=400, y=180)
    #side=RIGHT,padx=15, pady=15, ipadx=12, ipady=1)
    
    button2.pack()
    button2.place(x=35, y=360)
    #side=LEFT,padx=15, pady=15, ipadx=12, ipady=1)
    
    button1.pack(side=BOTTOM) #,padx=10, pady=1)

    # Display until closed manually
    top1.mainloop()


# Training Model Button
button1 = Button(text='Train Model', command=trainModel,
                 bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'))
canvas1.create_window(200, 100, window=button1)

# Trained Model Button
button2 = Button(text='Choose Trained Model', command=chooseModel,
                 bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'))
canvas1.create_window(200, 140, window=button2)

# CSV Selector Button
open_button = Button(root, text='Select Patient Record', command=predictSeizure,
                     bg='brown', fg='white', font=('Proxima Nova', 10, 'bold'))

# Display until closed manually
root.mainloop()
