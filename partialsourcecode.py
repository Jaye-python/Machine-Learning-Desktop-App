import matplotlib.pyplot as plt
import io
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib


def scaler():
    head = filter_heading.get()
    f = head.split(',')
    global ed
    scaler = preprocessing.MinMaxScaler()
    
     try:
        ped = ed[f]
        scaler.fit(ped)
        sed = pd.DataFrame(scaler.transform(ped), index=ed.index, columns=ped.columns)
        ed = ed.drop(f, axis=1)
        ed = ed.join(sed)
        extext.set(value= 'Export ' + str(len(ed)) + '\nrows')
        e.delete(1.0, END)
        e.insert(1.0, ed)
        stext.set(value='Scaled')
    except:
        messagebox.showinfo(' Info required', "Please indicate names of columns to be scaled")
        
 def selectkbest():
    head = filter_heading.get()
    f = head.split(',')

    global ed

    try:

        X = ed.drop(f, axis=1)
        y = ed[f]

        # selector = SelectKBest(score_func=chi2, k=3)
        # selector = SelectKBest(score_func=chi2, k=10)
        selector = SelectKBest(score_func=chi2, k='all')
        fit = selector.fit(X, y)

        scores = pd.DataFrame(fit.scores_)
        columns = pd.DataFrame(X.columns)

        featureScores = pd.concat([columns, scores],axis=1)
        featureScores.columns = ['Specs','Score']

        dp = featureScores.nlargest(15,'Score')

        # X_new = selector.transform(X)
        # p = X.columns[selector.get_support(indices=True)]
        # print(p)

        e.delete(1.0, END)
        e.insert(1.0, dp)
        sktext.set(value='K-Best Selected')

    except:
        messagebox.showinfo(' Info required', "Please indicate column name of 'X' variable")
   
  
  
def logisticRegression():
    global X
    global y
    global savealgorithm

    t0 = time.time()
    try:

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
        # clf_logreg = LogisticRegression()
        clf_logreg = LogisticRegression(max_iter=3000)
        # LogisticRegression(max_iter=2000)
        
        # clf_logreg.fit(X_train,y_train)
        clf_logreg.fit(X_train, y_train.values.ravel())

        savealgorithm = clf_logreg

        y_pred = clf_logreg.predict(X_test)

        # accuracy score
        score = round(accuracy_score(y_test, y_pred) * 100)
        scc = str(score)+'%'

        c_matrix = confusion_matrix(y_test, y_pred)
        
        time_taken = time.time()-t0

        cr = classification_report(y_test, y_pred)
        aentry.delete(1.0, END)
        aentry.insert(1.0, scc)

        e.delete(1.0, END)
        e.insert(1.0, cr)
        messagebox.showinfo(' Completed', "'Logistic Regression' algorithm executed. 'Accuracy Score' indicated above and 'Classification Report' indicated below in Output")
    except:
        messagebox.showinfo(' Info Required', 'Please create required models')
   
  def randomforestclassifier():
    global X
    global y
    global savealgorithm

    t0 = time.time()
    # DATA SPLICING
    
    try:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
        clf_rf = RandomForestClassifier(max_depth=4)
        # clf_rf = RandomForestClassifier()
        # clf_rf.fit(X_train, y_train)
        
        clf_rf.fit(X_train, y_train.values.ravel())

        savealgorithm = clf_rf
        
        y_pred = clf_rf.predict(X_test)
        score = round(accuracy_score(y_test, y_pred) * 100)
        scc = str(score)+'%'
        time_taken = time.time()-t0

        cr = classification_report(y_test, y_pred)
        aentry.delete(1.0, END)
        aentry.insert(1.0, scc)

        e.delete(1.0, END)
        e.insert(1.0, cr)
        messagebox.showinfo(' Completed', "'Random Forest Classifier' algorithm executed. 'Accuracy Score' indicated above and 'Classification Report' indicated below in Output") 
    except:
        messagebox.showinfo(' Info Required', 'Please create required models')
  
  
def predict():
    global ed
    # head = filter_heading.get()
    # f = head.split(',')
    try:
        model = joblib.load('algorithm.joblib')
        predict = model.predict(ed)
        # predict = model.predict([[ 21, 1 ], [22, 0 ]])
        predict = pd.DataFrame(predict)
        ed = ed.join(predict)
        extext.set(value= 'Export\n' + str(len(ed)) + '\nrows')
        e.delete(1.0, END)
        e.insert(1.0, ed)
        messagebox.showinfo(' Predicted', 'Prediction completed.')
    except:
        messagebox.showinfo(' Info required', "Please indicate algorithm, models and input features for prediction")
        
   
