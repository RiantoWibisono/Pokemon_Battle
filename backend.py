# ========================================================================================================
# Pokemon Battle
# ========================================================================================================
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, abort
import requests
import joblib
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import random

app = Flask(__name__, static_url_path='')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/hasil', methods = ['GET', 'POST'])
def hasil():
    df = pd.read_csv('pokemon.csv', index_col=0)
   
    pokemon1 = request.form['nama1']
    pokemon2 = request.form['nama2']

    if pokemon1.lower().title() in df['Name'].values:
        if pokemon2.lower().title() in df['Name'].values:
            feature = []
            dfPoke1 = df[df['Name'] == pokemon1.lower().title()]
            dfPoke2 = df[df['Name'] == pokemon2.lower().title()]

            feature.append(dfPoke1['HP'].values[0])
            feature.append(dfPoke2['HP'].values[0])
            feature.append(dfPoke1['Attack'].values[0])
            feature.append(dfPoke2['Attack'].values[0])
            feature.append(dfPoke1['Defense'].values[0])
            feature.append(dfPoke2['Defense'].values[0])
            feature.append(dfPoke1['Sp. Atk'].values[0])
            feature.append(dfPoke2['Sp. Atk'].values[0])
            feature.append(dfPoke1['Sp. Def'].values[0])
            feature.append(dfPoke2['Sp. Def'].values[0])
            feature.append(dfPoke1['Speed'].values[0])
            feature.append(dfPoke2['Speed'].values[0])
            
            prediction = int(model.predict([feature])[0])
            if prediction == 0:
                prediction = pokemon1
            elif prediction == 1:
                prediction = pokemon2
            probability = round(max(model.predict_proba([feature])[0]) * 100, 2)

            url = 'https://pokeapi.co/api/v2/pokemon/'
            dataPokemon1 = requests.get(url+pokemon1)
            dataPokemon2 = requests.get(url+pokemon2)

            plt.figure(figsize = (12,3))
            plt.style.use('ggplot')

            plt.subplot(161)
            plt.title('HP', size=10)
            plt.bar(pokemon1,df[df['Name'] == pokemon1.lower().title()]['HP'].values[0], color = 'b')
            plt.bar(pokemon2,df[df['Name'] == pokemon2.lower().title()]['HP'].values[0], color = 'g')

            plt.subplot(162)
            plt.title('Attack', size=10)
            plt.bar(pokemon1,df[df['Name'] == pokemon1.lower().title()]['Attack'].values[0], color = 'b')
            plt.bar(pokemon2,df[df['Name'] == pokemon2.lower().title()]['Attack'].values[0], color = 'g')

            plt.subplot(163)
            plt.title('Defense', size=10)
            plt.bar(pokemon1,df[df['Name'] == pokemon1.lower().title()]['Defense'].values[0], color = 'b')
            plt.bar(pokemon2,df[df['Name'] == pokemon2.lower().title()]['Defense'].values[0], color = 'g')

            plt.subplot(164)
            plt.title('Special Attack', size=10)
            plt.bar(pokemon1,df[df['Name'] == pokemon1.lower().title()]['Sp. Atk'].values[0], color = 'b')
            plt.bar(pokemon2,df[df['Name'] == pokemon2.lower().title()]['Sp. Atk'].values[0], color = 'g')

            plt.subplot(165)
            plt.title('Special Defense', size=10)
            plt.bar(pokemon1,df[df['Name'] == pokemon1.lower().title()]['Sp. Def'].values[0], color = 'b')
            plt.bar(pokemon2,df[df['Name'] == pokemon2.lower().title()]['Sp. Def'].values[0], color = 'g')

            plt.subplot(166)
            plt.title('Speed', size=10)
            plt.bar(pokemon1,df[df['Name'] == pokemon1.lower().title()]['Speed'].values[0], color = 'b')
            plt.bar(pokemon2,df[df['Name'] == pokemon2.lower().title()]['Speed'].values[0], color = 'g')

            xy = random.randint(10000, 9999999)
            listplot = os.listdir('./storage')
            aa = str(len(listplot) + 1) + '_' + str(xy) + '.jpg'

            plt.savefig('storage/%s' % aa, Transparent=True)

            return render_template('hasil.html', dataPokemon1 = dataPokemon1, dataPokemon2 = dataPokemon2, prediction = prediction, proba = probability, zz=aa)
        else:
            return render_template('error.html')
    else:
        return render_template('error.html')
    
@app.route('/plotku/<path:yy>')                                 
def plotku(yy):
    return send_from_directory('storage', yy)

@app.errorhandler(404)
def notFound(error):            
    return render_template('error.html'), 404

if __name__ == '__main__':
    model = joblib.load('ML.joblib')
    app.run(debug = True)