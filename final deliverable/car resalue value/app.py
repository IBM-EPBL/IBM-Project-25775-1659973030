import pickle

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request
from sklearn.preprocessing import LabelEncoder

app=Flask(__name__)
filename= 'resale.pkl'
model_rand = pickle.load(open(filename,'rb'))

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict')
def predict():
    return render_template('prediction.html')

@app.route('/y_predict',methods=['GET','POST'])
def y_predict():
    regyear = int(request.form['regyear'])
    powerps =float(request.form['powerps'])
    kms= float(request.form['kms'])
    regmonth = int(request.form.get('regmonth'))
    gearbox = request.form['gearbox']
    damage =request.form['dam']
    model = request.form.get('modeltype')
    brand = request.form.get('brand')
    fuelType = request.form.get('fuel')
    vehicletype = request.form.get('vehicletype')
    new_row = {'yearOfRegistration':regyear,'powerPs':powerps,'kilometer':kms,'monthOfRegistration':regmonth, 'gearbox':gearbox, 'notRepairedDamage':damage, 'model':model, 'brand':brand, 'fuelType':fuelType, 'vehicleType':vehicletype}
    print(new_row)
    new_df = pd.DataFrame(columns =['vehicleType','yearOfRegistration', 'gearbox', 'powerPs', 'model', 'kilometer', 'monthOfRegistration', 'fuleType', 'brand', 'notRepairedDamage'])
    new_df = new_df.append(new_row,ignore_index = True)
    labels = ['gearbox','notRepairedDamage','model','brand','fuelType','vehicleType']
    mapper={}
    for i in labels:
        mapper[i]=LabelEncoder()
        mapper[i].classes_=np.load(str('classes'+i+'.npy'))
        tr=mapper[i].fit_transform(new_df[i])
        new_df.loc[:, i +'labels'] = pd.Series(tr, index = new_df.index)
    labeled = new_df[['yearOfRegistration','powerPs','kilometer','monthOfRegistrstion']+[x+'_labels' for x in labels]]
    x=labeled.values
    print(x)
    y_prediction = model_rand.predict(x)
    print(y_prediction)
    return render_template('predict.html',ypred = 'The resale value predicted is {:.2f}$'.format(y_prediction[0]))
if __name__ == '__main__' :
    app.run(host='localhost',debug=True,threaded=False)

