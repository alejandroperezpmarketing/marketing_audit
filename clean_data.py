import pandas as pd
from ydata_profiling import ProfileReport
#%matplotlib inline
import matplotlib.pyplot as plt   #para el mapa
import matplotlib.image as mpimg
#algoritmos de optimizacion (aprender y predecir ej: tráfico) y recomendacion

### import data

pd.set_option('display.max_columns', None)
df = pd.read_csv('/home/vagrant/Documents/bigdata/machine_learning/data/1/telecom_customer_churn.csv',delimiter=',')
df = df.read()

### rename columns

df.rename(lambda x: x.lower().strip().replace(' ','_'),axis='columns',inplace=True)
df.columns

## profile report

report = ProfileReport(df,title='Profile Repport')
report.to_file('report.html')

def create_report(df=df, title='Profile Repport',file_path='./report.html'):
    report = ProfileReport(df,title=title)
    file_path = file_path
    report.to_file(file_path)
    return file_path

#create_report()


def assign_color(c):
    if c=='Churned': return 1
    elif c=='Stayed': return 2
    else: return 0

df2 = df[['longitude','latitude','customer_status']].copy()
df2['color'] = df['customer_status'].apply(lambda x: assign_color(x)).astype('category') # for each x in column the function will asign a code color category

def colorma(df,column_list,category_column,color_column_name):
    
    df2 = df[column_list].copy()
    df2[f'"{color_column_name}"'] = df2[f'"{category_column}"'].apply(lambda x: assign_color(x)).astype('category') 

    return 

    
    