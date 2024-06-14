#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:35:37 2024

@author: denisivan
"""

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

#Citim setul de date
df = pd.read_csv("laptop_price.csv", encoding='latin1')

#Aflam structura setului de date si informatii despre acesta
print(df.info())
print(df.describe())

#Stergem valorile nule
df.dropna(inplace=True)

#Verificam daca exista valori duplicate si in cazul in care exista le stergem(in caszul nostu avem 0 valori duplicate)
valori_duplicate = df.duplicated()
print("Numar de valori duplicate:", valori_duplicate.sum())

#Numarul total de inregistrari************
numar_inregistrari = df["laptop_ID"]
print ("Numar inregistrari: ", numar_inregistrari.count())

#Curatam coloana Product de explicatiile din paranteza
df['Product'] = df['Product'].str.split('(', n=1).str[0]

#Impartim ScreenResolution in doua coloane: PanelType si Resolution
df[['PanelType', 'Resolution']] = df['ScreenResolution'].str.extract(r'([A-Za-z\s/+|4K]+)?(\d+x\d+)?')

#Scoatem ScreenResolution
df = df.drop(["ScreenResolution"], axis=1)

#Inlocuim nan values cu valoarea cea mai frecventa care apare in coloana PanelType(mode)
df['PanelType'] = df['PanelType'].fillna(df['PanelType'].mode()[0])

#Impartim modelul de CPU si GHZ asociati acestuia
df[['Cpu', 'GHz']] = df['Cpu'].str.rsplit(" ", n=1, expand=True)

#Stergem sting-ul "GHz" din coloana GHz
df['GHz'] = df['GHz'].str.replace('GHz', '')

#Transformam in float coloana GHz
df['GHz'] = df['GHz'].astype(float)

#Stergem string-ul GB din coloana Ram
df['RamGB'] = df['Ram'].str.replace('GB', '')

#Transformam in int coloana Ram
df['RamGB'] = df['RamGB'].astype(int)

#Scoatem coloana Ram
df = df.drop(["Ram"], axis=1)

#Impartim producatorul de GPU si modelul asociat acestuia
df[['Gpu', 'Gpu_Model']] = df['Gpu'].str.split(" ", n=1, expand=True)

#Stergem string-ul kg din coloana Weight
df['WeightKG'] = df['Weight'].str.replace('kg', '')

#Transformam in float coloana Weight
df['WeightKG'] = df['WeightKG'].astype(float)

#Scoatem coloana Weight
df = df.drop(["Weight"], axis=1)

#Impartim Memory in doua coloane: MemoryGB si MemoryBonus
df[['MemoryGB', 'MemoryBonus']] = df['Memory'].str.split("+", n=1, expand=True)

#Scoatem GB
df['MemoryGB'] = df['MemoryGB'].str.replace('GB', '')

#Inlocuim 1TB cu 1000GB
df['MemoryGB'] = df['MemoryGB'].str.replace('1TB', '1000')

#Inlocuim 2TB cu 2000GB
df['MemoryGB'] = df['MemoryGB'].str.replace('2TB', '2000')

#Inlocuim 1.0TB cu 1000GB
df['MemoryGB'] = df['MemoryGB'].str.replace('1.0TB', '1000')

#Scoatem GB
df['MemoryBonus'] = df['MemoryBonus'].str.replace('GB', '')

#Inlocuim 1TB cu 1000GB
df['MemoryBonus'] = df['MemoryBonus'].str.replace('1TB', '1000')

#Inlocuim 2TB cu 2000GB
df['MemoryBonus'] = df['MemoryBonus'].str.replace('2TB', '2000')

#Inlocuim 1.0TB cu 1000GB
df['MemoryBonus'] = df['MemoryBonus'].str.replace('1.0TB', '1000')

#Cautam doar partea numerica din MemoryGB
df['MemoryGB'] = df['MemoryGB'].str.extract('(\d+)')

#Transformam valorile in float
df['MemoryGB'] = pd.to_numeric(df['MemoryGB'], errors='coerce')

#Cautam doar partea numerica din MemoryBonus
df['MemoryBonus'] = df['MemoryBonus'].str.extract('(\d+)')

#Transformam valorile in float
df['MemoryBonus'] = pd.to_numeric(df['MemoryBonus'], errors='coerce')

#Înlocuim valorile NaN cu 0************
df.fillna(0, inplace=True)

#Calcumam Memoria Totala
df['MemoryTotal'] = df['MemoryGB'] + df['MemoryBonus']

#Selectam string-urile care trebuie pastrate 
important_words = ['SSD', 'HDD',"Hybrid","Flash Storage"]

#Creem o expresie care sa contina toate combinatiile dintre toate cuvintele cheie
pattern = '|'.join(important_words)

#Extrage substring-ul potivit din coloana Memory
df['Memory'] = df['Memory'].str.extractall(f'({pattern})').groupby(level=0).agg(lambda x: '+'.join(x))

###############################################################################

#medii / cvartile/deviatie standard / mediana pt var. numerice ***********************

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "laptop_ID"
df_laptop_ID = pd.DataFrame({
    'Mean_laptop_ID': [df['laptop_ID'].mean()],
    'Std_laptop_ID': [df['laptop_ID'].std()],
    'Min_laptop_ID': [df['laptop_ID'].min()],
    'Quantile_25%_laptop_ID': [df['laptop_ID'].quantile(0.25)],
    'Quantile_50%_laptop_ID': [df['laptop_ID'].quantile(0.5)],
    'Quantile_75%_laptop_ID': [df['laptop_ID'].quantile(0.75)],
    'Max_laptop_ID': [df['laptop_ID'].max()]
})

# Afișăm rezultatul
print("Laptop_ID: ", df_laptop_ID)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "Inches"
df_Inches = pd.DataFrame({
    'Mean_Inches': [df['Inches'].mean()],
    'Std_Inches': [df['Inches'].std()],
    'Min_Inches': [df['Inches'].min()],
    'Quantile_25%_Inches': [df['Inches'].quantile(0.25)],
    'Quantile_50%_Inches': [df['Inches'].quantile(0.5)],
    'Quantile_75%_Inches': [df['Inches'].quantile(0.75)],
    'Max_Inches': [df['Inches'].max()]
})

# Afișăm rezultatul
print("Inches: ", df_Inches)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "Price_euros"
df_Price_euros = pd.DataFrame({
    'Mean_Price_euros': [df['Price_euros'].mean()],
    'Std_Price_euros': [df['Price_euros'].std()],
    'Min_Price_euros': [df['Price_euros'].min()],
    'Quantile_25%_Price_euros': [df['Price_euros'].quantile(0.25)],
    'Quantile_50%_Price_euros': [df['Price_euros'].quantile(0.5)],
    'Quantile_75%_Price_euros': [df['Price_euros'].quantile(0.75)],
    'Max_Price_euros': [df['Price_euros'].max()]
})

# Afișăm rezultatul
print("Price_euros: ", df_Price_euros)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "GHz"
df_GHz = pd.DataFrame({
    'Mean_GHz': [df['GHz'].mean()],
    'Std_GHz': [df['GHz'].std()],
    'Min_GHz': [df['GHz'].min()],
    'Quantile_25%_GHz': [df['GHz'].quantile(0.25)],
    'Quantile_50%_GHz': [df['GHz'].quantile(0.5)],
    'Quantile_75%_GHz': [df['GHz'].quantile(0.75)],
    'Max_GHz': [df['GHz'].max()]
})

# Afișăm rezultatul
print("GHz: ", df_GHz)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "RamGB"
df_RamGB = pd.DataFrame({
    'Mean_RamGB': [df['RamGB'].mean()],
    'Std_RamGB': [df['RamGB'].std()],
    'Min_RamGB': [df['RamGB'].min()],
    'Quantile_25%_RamGB': [df['RamGB'].quantile(0.25)],
    'Quantile_50%_RamGB': [df['RamGB'].quantile(0.5)],
    'Quantile_75%_RamGB': [df['RamGB'].quantile(0.75)],
    'Max_RamGB': [df['RamGB'].max()]
})

# Afișăm rezultatul
print("RamGB: ", df_RamGB)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "WeightKG"
df_WeightKG = pd.DataFrame({
    'Mean_WeightKG': [df['WeightKG'].mean()],
    'Std_WeightKG': [df['WeightKG'].std()],
    'Min_WeightKG': [df['WeightKG'].min()],
    'Quantile_25%_WeightKG': [df['WeightKG'].quantile(0.25)],
    'Quantile_50%_WeightKG': [df['WeightKG'].quantile(0.5)],
    'Quantile_75%_WeightKG': [df['WeightKG'].quantile(0.75)],
    'Max_WeightKG': [df['WeightKG'].max()]
})

# Afișăm rezultatul
print("WeightKG: ", df_WeightKG)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "MemoryGB"
df_MemoryGB = pd.DataFrame({
    'Mean_MemoryGB': [df['MemoryGB'].mean()],
    'Std_MemoryGB': [df['MemoryGB'].std()],
    'Min_MemoryGB': [df['MemoryGB'].min()],
    'Quantile_25%_MemoryGB': [df['MemoryGB'].quantile(0.25)],
    'Quantile_50%_MemoryGB': [df['MemoryGB'].quantile(0.5)],
    'Quantile_75%_MemoryGB': [df['MemoryGB'].quantile(0.75)],
    'Max_MemoryGB': [df['MemoryGB'].max()]
})

# Afișăm rezultatul
print("MemoryGB: ", df_MemoryGB)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "MemoryBonus"
df_MemoryBonus = pd.DataFrame({
    'Mean_MemoryBonus': [df['MemoryBonus'].mean()],
    'Std_MemoryBonus': [df['MemoryBonus'].std()],
    'Min_MemoryBonus': [df['MemoryBonus'].min()],
    'Quantile_25%_MemoryBonus': [df['MemoryBonus'].quantile(0.25)],
    'Quantile_50%_MemoryBonus': [df['MemoryBonus'].quantile(0.5)],
    'Quantile_75%_MemoryBonus': [df['MemoryBonus'].quantile(0.75)],
    'Max_MemoryBonus': [df['MemoryBonus'].max()]
})

# Afișăm rezultatul
print("MemoryBonus: ", df_MemoryBonus)

#Calculează media, deviația standard, mediana, min, max si cvartilele pentru coloana "MemoryTotal"
df_MemoryTotal = pd.DataFrame({
    'Mean_MemoryTotal': [df['MemoryTotal'].mean()],
    'Std_MemoryTotal': [df['MemoryTotal'].std()],
    'Min_MemoryTotal': [df['MemoryTotal'].min()],
    'Quantile_25%_MemoryTotal': [df['MemoryTotal'].quantile(0.25)],
    'Quantile_50%_MemoryTotal': [df['MemoryTotal'].quantile(0.5)],
    'Quantile_75%_MemoryTotal': [df['MemoryTotal'].quantile(0.75)],
    'Max_MemoryTotal': [df['MemoryTotal'].max()]
})

# Afișăm rezultatul
print("MemoryTotal: ", df_MemoryTotal)

###############################################################################

#Analiza exploratorie

#Cel mai popular panou
aparitie_panou = df['PanelType'].value_counts().idxmax()
print("Cel mai folosit Panou este: ", aparitie_panou)

#Cel mai popular Cpu
aparitie_Cpu = df['Cpu'].value_counts().idxmax()
print("Cel mai folosit Cpu este: ", aparitie_Cpu)

#Cel mai popular producator de Gpu
aparitie_Gpu = df['Gpu'].value_counts().idxmax()
print("Cel mai folosit producator de Gpu este: ", aparitie_Gpu)

#Cel mai popular Gpu
aparitie_Gpu_Model = df['Gpu_Model'].value_counts().idxmax()
print("Cel mai folosit Gpu este: ", aparitie_Gpu_Model)

#Cea mai populara rezolutie
aparitie_resolution = df['Resolution'].value_counts().idxmax()
print("Cea mai folosita rezolutie este: ", aparitie_resolution)

#Cel mai popular produs (cel mai cumparat)
aparitie_produs = df['Product'].value_counts().idxmax()
print("Cel mai folosit produs este: ", aparitie_produs)

#Cea mai populara companie
aparitie_companie = df['Company'].value_counts().idxmax()
print("Cea mai populara companie este: ", aparitie_companie)

#Cel mai popular sistem de operare
aparitie_opsys = df['OpSys'].value_counts().idxmax()
print("Cel mai folosit sistem de operare este: ", aparitie_opsys)

#Cea mai populara diagonala
aparitie_inci = df['Inches'].value_counts().idxmax()
print("Cea mai populara diagonala este: ", aparitie_inci)

#Cel mai nepopular panou
aparitie_panou_nepopular = df['PanelType'].value_counts().idxmin()
print("Cel mai nefolosit Panou este: ", aparitie_panou_nepopular)

#Cel mai nepopular Cpu
aparitie_Cpu_nepopular = df['Cpu'].value_counts().idxmin()
print("Cel mai nefolosit Cpu este: ", aparitie_Cpu_nepopular)

#Cel mai nepopular producator de Gpu
aparitie_Gpu_nepopular = df['Gpu'].value_counts().idxmin()
print("Cel mai nefolosit producator de Gpu este: ", aparitie_Gpu_nepopular)

#Cel mai nepopular Gpu
aparitie_Gpu_Model_nepopular = df['Gpu_Model'].value_counts().idxmin()
print("Cel mai nefolosit Gpu este: ", aparitie_Gpu_Model_nepopular)

#Cea mai nepopulara rezolutie
aparitie_resolution_nepopulara = df['Resolution'].value_counts().idxmin()
print("Cea mai nefolosita rezolutie este: ", aparitie_resolution_nepopulara)

#Cel mai nepopular produs (cel mai cumparat)
aparitie_produs_nepopular = df['Product'].value_counts().idxmin()
print("Cel mai nefolosit produs este: ", aparitie_produs_nepopular)

#Cea mai nepopulara companie
aparitie_companie_nepopulara = df['Company'].value_counts().idxmin()
print("Cea mai nepopulara companie este: ", aparitie_companie_nepopulara)

#Cel mai nepopular sistem de operare
aparitie_opsys_nepopular = df['OpSys'].value_counts().idxmin()
print("Cel mai nefolosit sistem de operare este: ", aparitie_opsys_nepopular)

#Cea mai nepopulara diagonala
aparitie_inci_nepopulara = df['Inches'].value_counts().idxmin()
print("Cea mai nepopulara populara diagonala este: ", aparitie_inci_nepopulara)

#Pret laptop pe fiecare companie + Grafic
pret_companie = df.groupby('Company')['Price_euros'].mean()
print('Media de pret pentru fiecare companie: ',pret_companie)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_companie.plot(kind='bar', color='skyblue')
plt.title('Pretul mediu in functie de companie')
plt.xlabel('Companie')
plt.ylabel('Pret mediu (euro)')
plt.xticks(rotation=45, ha='right')  # Rotirea etichetelor pentru a le face mai usor de citit
plt.show()

#Pret maxim laptop pe fiecare companie+ Grafic
pret_companie_max = df.groupby('Company')['Price_euros'].max()
print('Pret maxim laptop pe fiecare companie: ',pret_companie_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_companie_max.plot(kind='bar', color='green')
plt.title('Pretul maxim in functie de companie')
plt.xlabel('Companie')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare companie+ Grafic
pret_companie_min = df.groupby('Company')['Price_euros'].min()
print('Pret minim laptop pe fiecare comapanie: ',pret_companie_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_companie_min.plot(kind='bar', color='red')
plt.title('Pretul minim in functie de companie')
plt.xlabel('Companie')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare tip de laptop + Grafic
pret_type = df.groupby('TypeName')['Price_euros'].mean()
print('Pret laptop pe fiecare tip de laptop:',pret_type)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_type.plot(kind='bar', color='orange')
plt.title('Pretul in functie de tipul laptopului')
plt.xlabel('Tipul laptopului')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare tip de laptop + Grafic
pret_type_min = df.groupby('TypeName')['Price_euros'].min()
print('Pret minim laptop pe fiecare tip de laptop:',pret_type_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_type_min.plot(kind='bar', color='brown')
plt.title('Pretul minim in functie de tipul laptopului')
plt.xlabel('Tipul laptopului')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare tip de laptop + Grafic
pret_type_max = df.groupby('TypeName')['Price_euros'].max()
print('Pret maxim laptop pe fiecare tip de laptop:',pret_type_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_type_max.plot(kind='bar', color='darkorange')
plt.title('Pretul maxim in functie de tipul laptopului')
plt.xlabel('Tipul laptopului')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare tip de panou + Grafic
pret_panel = df.groupby('PanelType')['Price_euros'].mean()
print('Pret laptop pe fiecare tip de panou: ',pret_panel)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_panel.plot(kind='bar', color='yellow')
plt.title('Pretul in functie de tipul panoului')
plt.xlabel('Tipul panoului')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare tip de panou + Grafic
pret_panel_min = df.groupby('PanelType')['Price_euros'].min()
print('Pret minim laptop pe fiecare tip de panou: ',pret_panel_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_panel_min.plot(kind='bar', color='skyblue')
plt.title('Pretul minim in functie de tipul panoului')
plt.xlabel('Tipul panoului')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare tip de panou + Grafic
pret_panel_max = df.groupby('PanelType')['Price_euros'].max()
print('Pret maxim laptop pe fiecare tip de panou: ',pret_panel_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_panel_max.plot(kind='bar', color='red')
plt.title('Pretul maxim in functie de tipul panoului')
plt.xlabel('Tipul panoului')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare tip de rezolutie + Grafic
pret_resolution = df.groupby('Resolution')['Price_euros'].mean()
print('Pret laptop pe fiecare tip de rezolutie: ',pret_resolution)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_resolution.plot(kind='bar', color='black')
plt.title('Pretul in functie de tipul rezolutiei')
plt.xlabel('Tipul rezolutiei')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare tip de rezolutie + Grafic
pret_resolution_min = df.groupby('Resolution')['Price_euros'].min()
print('Pret minim laptop pe fiecare tip de rezolutie: ',pret_resolution_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_resolution_min.plot(kind='bar', color='red')
plt.title('Pretul minim in functie de tipul rezolutiei')
plt.xlabel('Tipul rezolutiei')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare tip de rezolutie + Grafic
pret_resolution_max = df.groupby('Resolution')['Price_euros'].max()
print('Pret maxim laptop pe fiecare tip de rezolutie: ',pret_resolution_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_resolution_max.plot(kind='bar', color='grey')
plt.title('Pretul maxim in functie de tipul rezolutiei')
plt.xlabel('Tipul rezolutiei')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare tip de Cpu + Grafic
pret_Cpu = df.groupby('Cpu')['Price_euros'].mean()
print('Pret laptop pe fiecare tip de Cpu: ',pret_Cpu)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Cpu.plot(kind='bar', color='lightblue')
plt.title('Pretul in functie de tipul Cpu')
plt.xlabel('Tipul Cpu')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare tip de Cpu + Grafic
pret_Cpu_min = df.groupby('Cpu')['Price_euros'].min()
print('Pret minim laptop pe fiecare tip de Cpu: ',pret_Cpu_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Cpu_min.plot(kind='bar', color='darkblue')
plt.title('Pretul minim in functie de tipul Cpu')
plt.xlabel('Tipul Cpu')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare tip de Cpu + Grafic
pret_Cpu_max = df.groupby('Cpu')['Price_euros'].max()
print('Pret maxim laptop pe fiecare tip de Cpu: ',pret_Cpu_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Cpu_max.plot(kind='bar', color='blue')
plt.title('Pretul maxim in functie de tipul Cpu')
plt.xlabel('Tipul Cpu')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare tip de Gpu + Grafic
pret_Gpu = df.groupby('Gpu')['Price_euros'].mean()
print('Pret laptop pe fiecare tip de Gpu:',pret_Gpu)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Gpu.plot(kind='bar', color='darkred')
plt.title('Pretul in functie de tipul Gpu')
plt.xlabel('Tipul Gpu')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare tip de Gpu + Grafic
pret_Gpu_min = df.groupby('Gpu')['Price_euros'].min()
print('Pret minim laptop pe fiecare tip de Gpu:',pret_Gpu_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Gpu_min.plot(kind='bar', color='pink')
plt.title('Pretul minim in functie de tipul Gpu')
plt.xlabel('Tipul Gpu')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare tip de Gpu + Grafic
pret_Gpu_max = df.groupby('Gpu')['Price_euros'].max()
print('Pret maxim laptop pe fiecare tip de Gpu:',pret_Gpu_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Gpu_max.plot(kind='bar', color='red')
plt.title('Pretul maxim in functie de tipul Gpu')
plt.xlabel('Tipul Gpu')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare tip de Ram + Grafic
pret_Ram = df.groupby('RamGB')['Price_euros'].mean()
print('Pret laptop pe fiecare tip de Ram: ',pret_Ram)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Ram.plot(kind='bar', color='pink')
plt.title('Pretul in functie de tipul Ram')
plt.xlabel('Tipul Ram')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare tip de Ram + Grafic
pret_Ram_min = df.groupby('RamGB')['Price_euros'].min()
print('Pret minim laptop pe fiecare tip de Ram: ',pret_Ram_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Ram_min.plot(kind='bar', color='brown')
plt.title('Pretul minim in functie de tipul Ram')
plt.xlabel('Tipul Ram')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare tip de Ram + Grafic
pret_Ram_max = df.groupby('RamGB')['Price_euros'].max()
print('Pret maxim laptop pe fiecare tip de Ram: ',pret_Ram_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_Ram_max.plot(kind='bar', color='orange')
plt.title('Pretul maxim in functie de tipul Ram')
plt.xlabel('Tipul Ram')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare in functie de Memorie + Grafic
pret_memorytotal = df.groupby('MemoryTotal')['Price_euros'].mean()
print('Pret laptop pe fiecare in functie de Memorie: ',pret_memorytotal)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_memorytotal.plot(kind='bar', color='salmon')
plt.title('Pretul in functie de memorie')
plt.xlabel('Memorie')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare in functie de Memorie + Grafic
pret_memorytotal_min = df.groupby('MemoryTotal')['Price_euros'].min()
print('Pret minim laptop pe fiecare in functie de Memorie: ',pret_memorytotal_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_memorytotal_min.plot(kind='bar', color='purple')
plt.title('Pretul minim in functie de memorie')
plt.xlabel('Memorie')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare in functie de Memorie + Grafic
pret_memorytotal_max = df.groupby('MemoryTotal')['Price_euros'].max()
print('Pret maxim laptop pe fiecare in functie de Memorie: ',pret_memorytotal_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_memorytotal_max.plot(kind='bar', color='black')
plt.title('Pretul maxim in functie de memorie')
plt.xlabel('Memorie')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pt fiecare tip de memorie
pret_memory = df.groupby('Memory')['Price_euros'].mean()
print("Pret laptop pt fiecare tip de memorie:", pret_memory)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_memory.plot(kind='bar', color='grey')
plt.title('Pretul in functie de tipul de memorie')
plt.xlabel('Memorie')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pt fiecare tip de memorie
pret_memory_min = df.groupby('Memory')['Price_euros'].min()
print("Pret minim laptop pt fiecare tip de memorie:", pret_memory_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_memory_min.plot(kind='bar', color='brown')
plt.title('Pretul minim in functie de tipul de memorie')
plt.xlabel('Memorie')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pt fiecare tip de memorie
pret_memory_max = df.groupby('Memory')['Price_euros'].max()
print("Pret laptop pt fiecare tip de memorie:", pret_memory_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_memory_max.plot(kind='bar', color='purple')
plt.title('Pretul maxim in functie de tipul de memorie')
plt.xlabel('Memorie')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare in functie de Operating Sistem + Grafic
pret_opsys = df.groupby('OpSys')['Price_euros'].mean()
print("Pret laptop pe fiecare in functie de Operating Sistem:", pret_opsys)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_opsys.plot(kind='bar', color='red')
plt.title('Pretul in functie de sistemul de operare')
plt.xlabel('Sistem Operare')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare in functie de Operating Sistem + Grafic
pret_opsys_min = df.groupby('OpSys')['Price_euros'].min()
print("Pret minim laptop pe fiecare in functie de Operating Sistem:", pret_opsys_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_opsys_min.plot(kind='bar', color='grey')
plt.title('Pretul minim in functie de sistemul de operare')
plt.xlabel('Sistem Operare')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare in functie de Operating Sistem + Grafic
pret_opsys_max = df.groupby('OpSys')['Price_euros'].max()
print("Pret maxim laptop pe fiecare in functie de Operating Sistem:", pret_opsys_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
pret_opsys_max.plot(kind='bar', color='red')
plt.title('Pretul maxim in functie de sistemul de operare')
plt.xlabel('Sistem Operare')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret laptop pe fiecare in functie de greutate + Grafic
pret_weight = df.groupby('WeightKG')['Price_euros'].mean()
print("Pret laptop pe fiecare in functie de greutate:", pret_weight)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
plt.scatter(pret_weight.index, pret_weight.values, color='purple')
plt.title('Pretul in functie de greutate')
plt.xlabel('Greutate')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret minim laptop pe fiecare in functie de greutate + Grafic
pret_weight_min = df.groupby('WeightKG')['Price_euros'].min()
print("Pret minim laptop pe fiecare in functie de greutate:", pret_weight_min)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
plt.scatter(pret_weight_min.index, pret_weight_min.values, color='brown')
plt.title('Pretul minim in functie de greutate')
plt.xlabel('Greutate')
plt.ylabel('Pret minim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Pret maxim laptop pe fiecare in functie de greutate + Grafic
pret_weight_max = df.groupby('WeightKG')['Price_euros'].max()
print("Pret maxim laptop pe fiecare in functie de greutate:", pret_weight_max)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
plt.scatter(pret_weight_max.index, pret_weight_max.values, color='black')
plt.title('Pretul maxim in functie de greutate')
plt.xlabel('Greutate')
plt.ylabel('Pret maxim (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

#Cate laptopuri pt fiecare companie
laptop_model = df.groupby('Company')['Product'].count()
print("Cate laptopuri pt fiecare companie:", laptop_model)

# Crearea unui grafic bar
plt.figure(figsize=(12, 6))
laptop_model.plot(kind='bar', color='orange')
plt.title('Laptopuri pentru fiecare companie')
plt.xlabel('Companie')
plt.ylabel('Numar')
plt.xticks(rotation=45, ha='right')
plt.show()

#Cel mai cumparat produs pentru fiecare companie
cumparat = df.groupby(['Company', 'Product']).size().reset_index(name='Number')
print("Cel mai cumparat produs pentru fiecare companie: ", cumparat)

#Crearea unui grafic bar
plt.figure(figsize=(12, 6))
cumparat.plot(kind='bar', color='purple')
plt.title('Cel mai cumparat produs')
plt.xlabel('Produs')
plt.ylabel('Numar')
plt.xticks(rotation=45, ha='right')
plt.show()

#Sistem operare in functie de Ram + Grafic
op_ram = df.groupby('OpSys')['RamGB'].mean().round()
print("Cat ram e nevoie pentru fiecare sistem de operare: ", op_ram)

#Crearea unui grafic bar
plt.figure(figsize=(12, 6))
op_ram.plot(kind='bar', color='skyblue')
plt.title('Ram pentru fiecare sistem de operare')
plt.xlabel('Sistem Operare')
plt.ylabel('Ram')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea celor mai scumpe 5 laptopuri
laptop5 = df.nlargest(5, 'Price_euros')[['Product', 'Price_euros']]
print("Top 5 cele mai scumpe laptopuri:", laptop5)

# Crearea unui grafic bar pentru cele mai scumpe laptopuri
plt.figure(figsize=(12, 6))
plt.scatter(laptop5['Product'], laptop5['Price_euros'], color='salmon')
plt.title('Cele mai scumpe 5 laptopuri')
plt.xlabel('Modelul laptopului')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea celor mai ieftine 5 laptopuri
laptop5s = df.nsmallest(5, 'Price_euros')[['Product', 'Price_euros']]
print("Top 5 cele mai ieftine laptopuri:", laptop5s)

# Crearea unui grafic bar pentru cele mai scumpe laptopuri
plt.figure(figsize=(12, 6))
plt.scatter(laptop5s['Product'], laptop5s['Price_euros'], color='purple')
plt.title('Cele mai ieftine 5 laptopuri')
plt.xlabel('Modelul laptopului')
plt.ylabel('Pret (euro)')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea top 5 laptopuri cu cea mai multa memorie
laptop5m = df.nlargest(5, 'MemoryTotal')[['Product', 'MemoryTotal']]
print("Top 5 laptopuri cu cea mai multa memorie:", laptop5m)

# Crearea unui grafic
plt.figure(figsize=(12, 6))
plt.scatter(laptop5m['Product'], laptop5m['MemoryTotal'], color='green')
plt.title('5 laptopuri cea mai multa memorie')
plt.xlabel('Modelul laptopului')
plt.ylabel('Memorie')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea top 5 laptopuri cu cea mai putina memorie
laptop5ms = df.nsmallest(5, 'MemoryTotal')[['Product', 'MemoryTotal']]
print("Top 5 laptopuri cu cea mai putina memorie:", laptop5ms)

# Crearea unui grafic
plt.figure(figsize=(12, 6))
plt.scatter(laptop5ms['Product'], laptop5ms['MemoryTotal'], color='orange')
plt.title('5 laptopuri cea mai putina memorie')
plt.xlabel('Modelul laptopului')
plt.ylabel('Memorie')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea top 5 laptopuri cu cel mai mult RAM
laptop5r = df.nlargest(5, 'RamGB')[['Product', 'RamGB']]
print("Top 5 laptopuri cu cel mai mult Ram:", laptop5r)

# Crearea unui grafic
plt.figure(figsize=(12, 6))
plt.scatter(laptop5r['Product'], laptop5r['RamGB'], color='skyblue')
plt.title('5 laptopuri cel mai mult RAM')
plt.xlabel('Modelul laptopului')
plt.ylabel('Ram')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea top 5 laptopuri cu cel mia putin RAM
laptop5rs = df.nsmallest(5, 'RamGB')[['Product', 'RamGB']]
print("Top 5 laptopuri cu cel mai putin Ram:", laptop5rs)

# Crearea unui grafic
plt.figure(figsize=(12, 6))
plt.scatter(laptop5rs['Product'], laptop5rs['RamGB'], color='purple')
plt.title('5 laptopuri cel mai putina RAM')
plt.xlabel('Modelul laptopului')
plt.ylabel('Ram')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea top 3 cele mai grele laptopuri
laptop3w = df.nlargest(3, 'WeightKG')[['Product', 'WeightKG']]
print("Top 3 cele mai grele laptopuri:", laptop3w)

# Crearea unui grafic
plt.figure(figsize=(12, 6))
plt.scatter(laptop3w['Product'], laptop3w['WeightKG'], color='red')
plt.title('3 laptopuri cele mai grele laptopuri')
plt.xlabel('Modelul laptopului')
plt.ylabel('Greutate')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișarea top 3 laptopuri cu cea mai mare diagonala
laptop5d = df.nlargest(3, 'Inches')[['Product', 'Inches']]
print("Top 5 laptopuri cu cea mai mare diagonala:", laptop5d)

# Crearea unui grafic
plt.figure(figsize=(12, 6))
plt.scatter(laptop5d['Product'], laptop5d['Inches'], color='purple')
plt.title('5 laptopuri cel mai mare diagonala')
plt.xlabel('Modelul laptopului')
plt.ylabel('Diagonala')
plt.xticks(rotation=45, ha='right')
plt.show()


###############################################################################
#Determinam coloanele categoriale
coloane_categoriale = df.select_dtypes(include=["object"]).columns.drop(['Company', 'Product', 'Cpu', 'Gpu', 'Memory', 'PanelType', 'Resolution', 'Gpu_Model'])

#Transformam coloanele folosind One Hot Encoding
df_encoded = pd.get_dummies(df, columns=coloane_categoriale, drop_first=True, dtype=int)

################################################################################ Pana aici am curatat df-ul*

# Selectează caracteristicile pentru regresie
features = df_encoded.drop(['Company', 'Product', 'Cpu', 'Gpu', 'Memory', 'PanelType', 'Resolution', 'Gpu_Model','Price_euros'], axis=1)

# Selectează variabila țintă
target = df_encoded['Price_euros']

# Împarte datele în seturi de antrenare și de testare
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

###############################################################################

#Corelatie df
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#Corelatie df_encoded
correlation_matrix_encoded = df_encoded.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_encoded, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

###############################################################################

#Coeficient Pearson
# Lista pentru a stoca rezultatele
correlation_results = []

# Calculați și adăugați coeficientul de corelație Pearson în listă
for feature in features.columns:
    correlation, _ = pearsonr(features[feature], target)
    correlation_results.append({'Feature': feature, 'Pearson Correlation': correlation})

# Creare DataFrame din lista de rezultate
correlation_df = pd.DataFrame(correlation_results)

# Afișare DataFrame cu rezultatele
print(correlation_df)

###############################################################################

#Regresie Liniara
model = LinearRegression()

model.fit(x_train, y_train)

y_prezis = model.predict(x_test)

r_patrat = model.score(x_train, y_train)    

scorul_r2 = r2_score(y_test, y_prezis)

print(f'Coeficientul de determinare (R^2): {scorul_r2}')

#Valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_prezis)
plt.xlabel('Preturi Reale')
plt.ylabel('Preturi Prezise')
plt.title('Preturi Reale vs. Preturi Prezise')
plt.show()

###############################################################################

#Regresie OLS
#Adauga o coloană constanta pentru termenul liber (intercept)
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

#Initializeaza si antreneaza modelul de regresie liniara folosind OLS
ols_model = sm.OLS(y_train, x_train).fit()

#Realizeaza predictii pe setul de testare
y_pred2 = ols_model.predict(x_test)

#Evaluează performanta modelului
r2rOLS = r2_score(y_test, y_pred2)

print(ols_model.summary())
print(f'Coeficientul de determinare (R^2): {r2rOLS}')

# Vizualizează valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred2)
plt.xlabel('Prețuri Reale')
plt.ylabel('Prețuri Prezise')
plt.title('Prețuri Reale vs. Prețuri Prezise (Regresie OLS)')
plt.show()

###############################################################################

#Regresie Random Forest
# Inițializați și antrenați Random Forest Regressor
rf_model_laptop = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model_laptop.fit(x_train, y_train)

# Make predictions on the test set
y_pred = rf_model_laptop.predict(x_test)

#Evaluam performanta folosind metrici precum coeficientul R^2
r2 = r2_score(y_test, y_pred)

print(f'Coeficientul de determinare R^2: {r2}')

# Creează un grafic scatter pentru regresia Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Regresie Random Forest: Valori Reale vs. Prezise')
plt.xlabel('Valori Reale')
plt.ylabel('Valori Prezise')
plt.show()

###############################################################################

#Ca la curs pt multicoliniaritate

#Importanta
feature_importance = rf_model_laptop.feature_importances_

coeficienti = ols_model.params
p_values = ols_model.pvalues

rez_regresie = pd.DataFrame({
    "Coeficient": coeficienti,
    "p-values": p_values
    })

r_squared = ols_model.rsquared

###############################################################################

#VIF - pt multicoliniaritate
x_with_const = sm.add_constant(features)
vif_data = pd.DataFrame()
vif_data["Variable"] = x_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(x_with_const.values, i) for i in range(x_with_const.shape[1])]

print(vif_data)

###############################################################################

#Optimizare set de date dupa rulare VIF
features2 = df_encoded.drop(['Company', 'Product', 'Cpu', 'Gpu', 'Memory', 'PanelType', 'Resolution', 'Gpu_Model','Price_euros','MemoryGB','MemoryBonus','OpSys_Linux','OpSys_Chrome OS','OpSys_No OS','OpSys_Windows 10','OpSys_Windows 7'], axis=1)

#VIF - pt multicoliniaritate
x_with_const = sm.add_constant(features2)
vif_data2 = pd.DataFrame()
vif_data2["Variable"] = x_with_const.columns
vif_data2["VIF"] = [variance_inflation_factor(x_with_const.values, i) for i in range(x_with_const.shape[1])]

print(vif_data2)

#RFE
rfe = RFE(estimator = model, n_features_to_select=10)
rf = rfe.fit(features2, target)

atribute_selectate = pd.DataFrame({
    "atribut": features2.columns,
    "selectate": rfe.support_,
    "ranking": rfe.ranking_
    })

###############################################################################

#Regresie pe modelul optimizat
atribute_selectate_pt_reg = atribute_selectate[atribute_selectate["selectate"]]["atribut"]

features_selected= features2[atribute_selectate_pt_reg]

x_tr_sel, x_test_sel, y_tr_sel, y_test_sel = train_test_split(features_selected, target, test_size = 0.2, random_state=0)

reg_lin_sel = LinearRegression()
reg_lin_sel.fit(x_tr_sel, y_tr_sel)

y_prezis_sel = reg_lin_sel.predict(x_test_sel)
r2_scor_selectat = r2_score(y_test_sel, y_prezis_sel)

print(r2_scor_selectat)

#Valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_sel, y=y_prezis_sel)
plt.xlabel('Preturi Reale')
plt.ylabel('Preturi Prezise')
plt.title('Preturi Reale vs. Preturi Prezise Selectate')
plt.show()

###############################################################################

#Regresie OLS selectat
#Adauga o coloană constanta pentru termenul liber (intercept) selectat
x_tr_sel = sm.add_constant(x_tr_sel)
x_test_sel = sm.add_constant(x_test_sel)

#Initializeaza si antreneaza modelul de regresie liniara folosind OLS selectat
ols_model_selectat = sm.OLS(y_tr_sel, x_tr_sel).fit()

#Realizeaza predictii pe setul de testare selectat
y_pred2_selectat = ols_model_selectat.predict(x_test_sel)

#Evaluează performanta modelului selectat
r2rOLS_selectat = r2_score(y_test_sel, y_pred2)

print(ols_model_selectat.summary())
print(f'Coeficientul de determinare (R^2): {r2rOLS}')

# Vizualizează valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_sel, y=y_pred2_selectat)
plt.xlabel('Prețuri Reale')
plt.ylabel('Prețuri Prezise')
plt.title('Prețuri Reale vs. Prețuri Prezise (Regresie OLS) Selectate')
plt.show()

###############################################################################

#Regresie Random Forest
# Inițializați și antrenați Random Forest Regressor selectat
rf_model_laptop_select = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model_laptop_select.fit(x_tr_sel, y_tr_sel)

# Make predictions on the test set selected
y_pred_select = rf_model_laptop_select.predict(x_test_sel)

#Evaluam performanta folosind metrici precum coeficientul R^2 selectat
r2_select = r2_score(y_test_sel, y_pred_select)

print(f'Coeficientul de determinare R^2: {r2_select}')

# Creează un grafic scatter pentru regresia Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test_sel, y_pred_select, alpha=0.5)
plt.title('Regresie Random Forest: Valori Reale vs. Prezise Selectate')
plt.xlabel('Valori Reale')
plt.ylabel('Valori Prezise')
plt.show()

