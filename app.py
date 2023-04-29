import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
ds=pd.read_csv(r"C:\Users\AYUSH VIT\Downloads\diabetes.csv")
X = ds[ds.columns[:-1]].values
y = ds['Outcome'].values
x=ds.iloc[:,:-1].values
Y=ds.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.20,random_state=0)
model = RandomForestClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("Accuracy for Random Forest Classifier is : ",accuracy_score(predictions,y_test))




import pickle
pickle.dump(model,open(r"C:\Users\AYUSH VIT\Downloads\model_saved",'wb'))
model_loaded=pickle.load(open(r"C:\Users\AYUSH VIT\Downloads\model_saved",'rb'))
model_loaded.predict(x_test)





from flask import Flask, request,render_template
import pickle
import numpy as np
app = Flask(__name__,template_folder=r'C:\Users\AYUSH VIT\Frontend tools\template',static_folder=r'C:\Users\AYUSH VIT\Frontend tools\static')
model_loaded=pickle.load(open(r"C:\Users\AYUSH VIT\Downloads\model_saved",'rb'))
@app.route('/')
def home():
    return render_template('diabe.html')

@app.route('/submit-form', methods=['GET','POST'])
def submit_form():
  int_features = [x for x in request.form.values()]
  final_features = [np.array(int_features)]
  prediction = model.predict(final_features)
  output = prediction[0]

  if output == 0:
      return render_template('diabe.html', prediction_text= 'You are safe')
 
  else:
      return render_template('diabe.html', prediction_text= 'You have chance of having diabetes')
   


if __name__ == "__main__":
    app.run(debug=True)