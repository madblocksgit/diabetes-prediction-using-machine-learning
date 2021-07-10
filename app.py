from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('diabetes_dataset_preprocessed.csv')

x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(accuracy_score(y_pred,y_test))

app=Flask(__name__)

@app.route('/')
def gets_connected():
 return(render_template('index.html'))

@app.route('/',methods=['POST'])
def read_data():
 print('Reading data from Form')
 age=int(request.form['age'])
 gender=int(request.form['gender'])
 fdiabetes=int(request.form['fdiabetes'])
 highbp=int(request.form['highbp'])
 pa=int(request.form['pa'])
 bmi=int(request.form['bmi'])
 smoke=int(request.form['smoke'])
 alcohol=int(request.form['alcohol'])
 soundsleep=int(request.form['soundsleep'])
 medicine=int(request.form['medicine'])
 junkfood=int(request.form['junkfood'])
 stress=int(request.form['stress'])
 bplevel=int(request.form['bplevel'])
 pdiabetes=0
 uriation=int(request.form['uriation'])
 pregnancies=int(request.form['pregnancies'])
 sleep=soundsleep
 print(age,gender,fdiabetes,highbp,pa,bmi,smoke,alcohol,soundsleep,soundsleep,medicine,junkfood,stress,bplevel,pregnancies,pdiabetes,uriation)
 text=classifier.predict(([[age,gender,fdiabetes,highbp,pa,bmi,smoke,alcohol,sleep,soundsleep,medicine,junkfood,stress,bplevel,pregnancies,pdiabetes,uriation]]))
 print(text)
 if text[0]==0:
  k='No'
 else:
  k='Yes'
 return(render_template('index.html' ,prediction_output=k))

if __name__=="__main__":
 app.run(debug=True)