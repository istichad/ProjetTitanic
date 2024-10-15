import pandas as pd
import re
from django.shortcuts import render
from django.http import HttpResponse
import os
from django.conf import settings
import joblib

# Charger le modèle enregistré
model_path = os.path.join(settings.BASE_DIR, 'predictions', 'rf_model.pkl')
model = joblib.load(model_path)
def preprocess_data(data):
    # Imprimer les données avant le traitement
    print("Données avant le prétraitement:")
    print(data)
    
    # Remplir les valeurs manquantes et encodage
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Cabin_known'] = data['Cabin'].notna().astype(int)
    data.drop('Cabin', axis=1, inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Réorganiser les colonnes
    expected_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin_known', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0  

    data = data.reindex(columns=expected_columns)

    # Imprimer les données après le traitement
    print("Données après le prétraitement:")
    print(data)
    
    return data



def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        # Récupérer les données du formulaire
        pclass = int(request.POST.get('pclass'))
        sexe = request.POST.get('sexe').lower() 
        age = float(request.POST.get('age'))
        sibsp = int(request.POST.get('FrereSoeur'))
        parch = int(request.POST.get('parent'))
        fare = float(request.POST.get('tarif'))
        embarked = request.POST.get('port')  
        cabin = request.POST.get('cabine')  
        
        # Créer un DataFrame avec les nouvelles données
        new_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sexe],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Cabin': [cabin],
            'Embarked': [embarked]
        })
       
        new_data_preprocessed = preprocess_data(new_data)

        new_prediction = model.predict(new_data_preprocessed)

        categories = {0: "La personne n'a pas survécu", 1: "La personne a survécu"}
        resultat = categories[new_prediction[0]]  

        return render(request, 'output.html', {'result': resultat})
    
    return render(request, 'index.html')
