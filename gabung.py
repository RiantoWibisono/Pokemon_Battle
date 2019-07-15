# ========================================================================================================
# Pokemon Battle
# ========================================================================================================
import numpy as np 
import pandas as pd 

df1 = pd.read_csv('combats.csv')
df2 = pd.read_csv('pokemon.csv', index_col = 0, usecols=['#', 'Name','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'])

# ------------------------------------------------------------
# File ini bertujuan untuk menggabungkan dataset dari combats.csv dan pokemon.csv, yang nantinya akan menjadi data input untuk model machine learning

id1 = []
id2 = []
hp1 = []
hp2 = []
atk1 = []
atk2 = []
def1 = []
def2 = []
spatk1 = []
spatk2 = []
spdef1 = []
spdef2 = []
speed1 = []
speed2 = []
winner = []

for i in range(len(df1)):
    id1.append(df1.iloc[i]['First_pokemon'])
    id2.append(df1.iloc[i]['Second_pokemon'])
    hp1.append(df2.loc[df1.iloc[i]['First_pokemon']]['HP'])
    hp2.append(df2.loc[df1.iloc[i]['Second_pokemon']]['HP'])
    atk1.append(df2.loc[df1.iloc[i]['First_pokemon']]['Attack'])
    atk2.append(df2.loc[df1.iloc[i]['Second_pokemon']]['Attack'])
    def1.append(df2.loc[df1.iloc[i]['First_pokemon']]['Defense'])
    def2.append(df2.loc[df1.iloc[i]['Second_pokemon']]['Defense'])
    spatk1.append(df2.loc[df1.iloc[i]['First_pokemon']]['Sp. Atk'])
    spatk2.append(df2.loc[df1.iloc[i]['Second_pokemon']]['Sp. Atk'])
    spdef1.append(df2.loc[df1.iloc[i]['First_pokemon']]['Sp. Def'])
    spdef2.append(df2.loc[df1.iloc[i]['Second_pokemon']]['Sp. Def'])
    speed1.append(df2.loc[df1.iloc[i]['First_pokemon']]['Speed'])
    speed2.append(df2.loc[df1.iloc[i]['Second_pokemon']]['Speed'])
    
    if df1.iloc[i]['Winner'] == df1.iloc[i]['First_pokemon']:
        win = 0
        winner.append(win)        
    else:
        win = 1
        winner.append(win)        

df = pd.DataFrame(
    dict(
        id1 = id1,
        id2 = id2,
        hp1 = hp1,
        hp2 = hp2,
        atk1 = atk1,
        atk2 = atk2,
        def1 = def1,
        def2 = def2,
        spatk1 = spatk1,
        spatk2 = spatk2,
        spdef1 = spdef1,
        spdef2 = spdef2,
        speed1 = speed1,
        speed2 = speed2,
        winner = winner
    ))

 
df.to_csv('dataGabungan.csv')