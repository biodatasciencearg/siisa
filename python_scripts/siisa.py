#Rutina que esconde el codigo para mayor legibilidad.
from IPython.display import HTML
import random
import  numpy as np
import pandas as pd
def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide Code'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)

# auxiliary functions.
def get_data_fsql(query,serverName='MiAdelanto'):
    """Traer data de las bases SQL
    query=String donde hago la query que quiero traer
    serverName= String(siisa/MiAdelanto)"""
    import pymssql
    import pandas as pd
    #Select Server 'MiAdelanto'
    if serverName=='MiAdelanto':
        #Name server.
        server='miadelanto.cw8tpboctrtb.us-west-2.rds.amazonaws.com'
        # user server.
        user='jesica'
        # Password.
        password='Jesisol123'
        # used Database.
        db='dtsMiAdelanto'
    #Select Server 'siisa'
    elif serverName=='siisa':
        #Name server.
        server='motorsiisa2.cn5dtopedl5u.us-east-1.rds.amazonaws.com'
        # user server.
        user='elias'
        # Password.
        password='123456'
        # used Database.
        db='sbrde'
    elif serverName=='MINISIISA':
        #Name server.
        server='190.221.2.4'
        # user server.
        user='felipe'
        # Password.
        password='f3l1p3'
        # used Database.
        db='MINISIISA'

        
    else:
        print("Base de datos seleccionada incorrecta:")
    # connector instance
    conn = pymssql.connect(server=server, user=user, password=password, database=db)
    # Selected Query.
    df = pd.read_sql(query, conn)
    # Close connection.
    conn.close()
    return df


# Funciones para calculo de metricas.
def get_vertex_sum_row(index,row):
    """Devuelve los valores de la diagonal +/- 1 sumados, para una dada fila"""
    # si es la primera fila:
    if index==0:
        r=row[index] +row[index+1]
    # La ultima fila.
    elif len(row)==index+1:
        r=row[index-1] + row[index]        
    # Todas las demas filas.
    else:
        r=row[index-1] + row[index] +row[index+1]
    return r

def get_vertex_sum_array(df):
    """Devuelve el array de valores de la diagonal +/- 1 de todo el df. """
    # Genero el array de vertices.
    vertex_array=[]
    # itero sobre filas/columnas.
    for index in range(0,len(df)) : 
        # seteo la fila.
        row =(df.iloc[index].values) 
        # agrego la suma de los vertices a un array
        vertex_array.append(get_vertex_sum_row(index,row))
    return vertex_array

def accuracyPlusOne(df):
    """Retorna el valor de pureza/Captura dependiendo de si veo columnas o filas.
    
    Ejemplo 
        get_pureza(df)     Retorna la pureza.
        get_pureza(df.T)     Retorna el recupero.
    Mientras en filas sea prediccion y columnas reales.
    """
    return(np.sum(get_vertex_sum_array(df)) / df.values.sum())

def sueldo2cat(sueldo):
    """Function that provides a tuple (idCat, Cat) from an int, expressed in thousands of pesos."""
    if 0 <= sueldo < 13:
        return (11, 'D2/E')
    elif 13 <= sueldo < 21:
        return (10, 'D1-I')
    elif 21 <= sueldo < 29:
        return (9, 'D1-S')
    elif 29 <= sueldo < 35:
        return (8, 'C3-I')
    elif 35 <= sueldo < 41:
        return (7, 'C3-M')
    elif 41 <= sueldo < 48:
        return (6, 'C3-S')
    elif 48 <= sueldo < 59:
        return (5, 'C2-I')  
    elif 59 <= sueldo < 74:
        return (4, 'C2-M')
    elif 74 <= sueldo < 87:
        return (3, 'C2-S')
    elif 87 <= sueldo < 105:
        return (2, 'C1 55')
    elif 105 <= sueldo:
        return (1, 'AB')
    else:
        return "Error must provide a int between 0 and inf."   
def make_siisaPlot(df):
    """Funcion que retorna algunos plots interesantes del df de variables de siisa."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    df['scorecardclass']=df.scorecard.apply(lambda x: 'HIT' if (x <= 5) else 'THIN')
    plt.figure(figsize=(10,10))
    plt.subplot(3, 3, 1)
    # Distribucion de sexo.
    ax=df.sexo.apply(lambda x: 'MUJER' if (x==2) else ('HOMBRE' if (x==1) else 'I')).value_counts().plot(kind='pie',autopct='%.2f%%', labels=['','','',''],  fontsize=10)
    ax.legend(loc=3, labels=df.sexo.apply(lambda x: 'MUJER' if (x==2) else ('HOMBRE' if (x==1) else 'I')).value_counts().index)
    # Distribucion de autonomos.
    plt.subplot(3, 3, 2)
    ax2=df.autonomo.apply(lambda x: 'SI' if (x==1) else 'NO' ).value_counts().plot(kind='pie',autopct='%.2f%%', labels=['','','',''],  fontsize=10)
    ax2.legend(loc=3, labels=df.autonomo.apply(lambda x: 'SI' if (x==1) else 'NO' ).value_counts().index)
    # Distribucion de relacion de dependecia.
    plt.subplot(3, 3, 3)
    ax3=df.relac_dep.apply(lambda x: 'SI' if (x==1) else 'NO' ).value_counts().plot(kind='pie',autopct='%.2f%%', labels=['','','',''],  fontsize=10)
    ax3.legend(loc=3, labels=df.relac_dep.apply(lambda x: 'SI' if (x==1) else 'NO' ).value_counts().index)
    # Distribucion de scoreCard.
    plt.subplot(3, 3, 4)
    ax4=df.scorecardclass.value_counts().plot(kind='pie',autopct='%.2f%%', labels=['','','',''],  fontsize=10)
    ax4.legend(loc=3, labels=df.scorecardclass.value_counts().index)
    # Distribucion de Mora SIISA.
    plt.subplot(3, 3, 5)
    ax3=df.morasiisahist.apply(lambda x: 'SI' if (x==1) else 'NO' ).value_counts().plot(kind='pie',autopct='%.2f%%', labels=['','','',''],  fontsize=10)
    ax3.legend(loc=3, labels=df.morasiisahist.apply(lambda x: 'SI' if (x==1) else 'NO' ).value_counts().index)
    # Distribucion de Scores.
    plt.subplot(3, 3, 6)
    ax5=sns.distplot(df[df.score.notnull()].score,kde=False,norm_hist=True)
    ax5.axvline(x=700,color='red')
    
def get_l_borrar(columnsL):
    """funcion para generar las listas de columnas a borrar para el SES"""
    lista_a_borrar = ['antiguedad', 'autonomo','subsectorpc','relac_dep','jubilado']
    lista_a_borrar_final=[]
    for e in lista_a_borrar:
        l_ = [s for s in columnsL if e in s]
        lista_a_borrar_final += l_
    return lista_a_borrar_final


    
def cleanDf(df2,idcat=False,ses=False):
    """Funcion para limpiar el df con las variables que me vengan del score 2019.
    uso:
    cleanDf(df,idcat=True/False)  para que ponga las etiquetas de idcat sobre la columna del mismo nombre. defecto=False   
    """
    df=df2.copy()
    # paso todo a lower
    df.columns = [str.lower(x) for x in df.columns]
    # borro algunas columnas.
    if 'fechaactual' in df.columns:
        del df['fechaactual']
    if 'fecha' in df.columns:
        del df['fecha']
    #del df['nroconsulta']
    if 'cuil' in df.columns:
        del df['cuil']
    if 'nrodoc' in df.columns:
        del df['nrodoc']
    # Saco los de edad mayor a 100.
    df=df[df.age<100]
    # valores de sexo. 
    sexo_dummies= pd.get_dummies(df.sexo,prefix='sexo')
    # Agrego la dummy sexo.
    df = pd.concat([df,sexo_dummies], axis=1)
    # borro algunas columnas.
    if idcat:
        # Genero una columna agregando el idCat
        df['idcat']=df.ingreso.apply(lambda x: sueldo2cat(x)[0])
    if ses:
        # borro las columnas que no necesito para SES.
        for c in get_l_borrar(df.columns):
            del df[c]
        del df['score']
        del df['ingreso']
        del df['sexo']
    # saco columnas con mas del 80 % de nans.
    df = df.dropna(thresh=int(df.shape[0]*0.20), axis=1)
    # Imputo algunos campos de manera inteligente.
    values = {'mesesultimamora':999,'c_wcurr': 0, 'b_new': 9999, 'b_old': 9999, 'capitaltotalmora': 0, 'tendenciamonto1s6m' :0, 'tendenciamonto1s6m_r_sq' :0, 'tendenciamonto1s12m':0,'tendenciamonto1s12m_r_sq':0,'tendenciamonto1s24m':0,'tendenciamonto1s24m_r_sq':0, 'maxbancarizacion':0, 'mesesbancarizacion_banco':0,'mesesbancarizacion_norm_scorecard3_nofinanciera':0,'mesesBancarizacion_norm_scorecard3_banco':0,'mesesbancarizacion_banco':0,'mesesbancarizacion':0,'mesesbancarizacion_norm_scorecard3_banco':0} 
    df.fillna(value=values,inplace=True)
    

    return df

def get_pureza_captura(df,pureza=True):
    """Devuelve el array de valores de la diagonal +/- 1 de todo el df. """
    df_tmp=df.copy
    if pureza:
        df_tmp=df.T
    # Genero el array de vertices.
    pureza_captura_array=[]
    # itero sobre filas/columnas.
    for index in range(0,len(df)) : 
        # seteo la fila.
        row =(df_tmp.iloc[index].values) 
        # agrego la suma de los vertices a un array
        pureza_captura_array.append((round(((get_vertex_sum_row(index,row)/np.sum(row))*100),2)))
    return pureza_captura_array


def accuracyPlusOneDf(df):
    """Rutina que devuelve el accuracy y algunas metricas derivadas."""
    df_tmp=df.copy()
    captura_array=get_pureza_captura(df_tmp)
    pureza_array=get_pureza_captura(df_tmp.T)
    dicc={}
    for i,c in enumerate(df_tmp.columns):
        dicc[c]= [captura_array[i]]
    df_tmp['Cap/Pur']=pd.DataFrame(pureza_array,index=df_tmp.index)
    df_tmp=df_tmp.append(pd.DataFrame(dicc),sort=False)
    print("accuracyPlusOne:",round((np.sum(get_vertex_sum_array(df)) / df.values.sum()),5))
    return df_tmp

def get_metrics_accuracy(model,X_test,y_test,le):
    """Funcion para devolver las metricas de interes
    model es el objeto creado que tiene el modelo entrenado(ensambles)
    X_test vector de caracteristicas para  testeo
    y_test vector objetivo de testeo.
    le objeto del label encoder """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    y_predicted= model.predict(X_test)
    conf=confusion_matrix(y_predicted,y_test)
    predicted_cols = ['pred_'+ str(c) for c in le.classes_]
    cm=pd.DataFrame(conf, index = predicted_cols , columns = le.classes_)
    print('accuracy:',accuracy_score(y_predicted,y_test))
    return (accuracyPlusOneDf(cm))

def get_pureza_captura_plot(modelList,le):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    # defino df de captura y otro de pureza.
    df_pureza= pd.DataFrame()
    df_captura= pd.DataFrame()
    # itero sobre los modelos.
    for modeltuples in modelList:
        y_predicted=modeltuples[0]
        y_test=modeltuples[1]
        modelname=modeltuples[2]
        conf=confusion_matrix(y_predicted,y_test)
        predicted_cols = ['pred_'+ str(c) for c in le.classes_]
        cm=pd.DataFrame(conf, index = predicted_cols , columns = le.classes_)
        # Using DataFrame.insert() to add a column 
        df_pureza.insert(0, modelname, get_pureza_captura(cm,pureza=True), True) 
        df_captura.insert(0, modelname, get_pureza_captura(cm.T), True) 
    return(df_pureza,df_captura)
hide_toggle()


def get_ks(df,column_score_name,nbins=10):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    data=df.copy()
    data['good'] = 1 - data.bad
    data['bucket'] = pd.qcut(data[column_score_name], nbins)
    # GROUP THE DATA FRAME BY BUCKETS
    grouped = data.groupby('bucket', as_index = False)
    # CREATE A SUMMARY DATA FRAME
    agg1 = pd.DataFrame(grouped.min()[column_score_name])
    agg1.columns=['min_scr']
    agg1['max_scr'] = grouped.max()[column_score_name]
    agg1['bads'] = grouped.sum().bad
    agg1['goods'] = grouped.sum().good
    agg1['total'] = agg1.bads + agg1.goods
    
    # SORT THE DATA FRAME BY SCORE
    agg2 = (agg1.sort_values(by = 'min_scr')).reset_index(drop = True)
    agg2['cumsum_bads']= np.round(((agg2.bads / data.bad.sum()).cumsum()), 4) * 100
    agg2['cumsum_goods']=np.round(((agg2.goods / data.good.sum()).cumsum()), 4) * 100
    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
    
    # CALCULATE KS STATISTIC
    agg2['ks'] = np.round(((agg2.bads / data.bad.sum()).cumsum() - (agg2.goods / data.good.sum()).cumsum()), 4) * 100
    # DEFINE A FUNCTION TO FLAG MAX KS
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
    # FLAG OUT MAX KS
    agg2['max_ks'] = agg2.ks.apply(flag)
    print(agg2.ks.mean())
    plt.plot(list(range(1,(nbins+1))),agg2.cumsum_bads)
    plt.plot(list(range(1,(nbins+1))),agg2.cumsum_goods)
    return agg2

def get_ks_max(df,column_score_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    data=df.copy()
    data['good'] = 1 - data.bad
    data['bucket'] = pd.qcut(data[column_score_name], 10)
    # GROUP THE DATA FRAME BY BUCKETS
    grouped = data.groupby('bucket', as_index = False)
    # CREATE A SUMMARY DATA FRAME
    agg1 = pd.DataFrame(grouped.min()[column_score_name])
    agg1.columns=['min_scr']
    agg1['max_scr'] = grouped.max()[column_score_name]
    agg1['bads'] = grouped.sum().bad
    agg1['goods'] = grouped.sum().good
    agg1['total'] = agg1.bads + agg1.goods
    
    # SORT THE DATA FRAME BY SCORE
    agg2 = (agg1.sort_values(by = 'min_scr')).reset_index(drop = True)
    agg2['cumsum_bads']= np.round(((agg2.bads / data.bad.sum()).cumsum()), 4) * 100
    agg2['cumsum_goods']=np.round(((agg2.goods / data.good.sum()).cumsum()), 4) * 100
    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
    
    # CALCULATE KS STATISTIC
    agg2['ks'] = np.round(((agg2.bads / data.bad.sum()).cumsum() - (agg2.goods / data.good.sum()).cumsum()), 4) * 100
    # FLAG OUT MAX KS
    return (agg2.ks.max(),agg2.ks.mean())

def PermImportance(X, y, clf, metric, num_iterations=100):
    '''
    Calculates the permutation importance of features in a dataset.
    Inputs:
    X: dataframe with all the features
    y: array-like sequence of labels
    clf: sklearn classifier, already trained on training data
    metric: sklearn metric, such as accuracy_score, precision_score or recall_score
    num_iterations: no. of repetitive runs of the permutation
    Outputs:
    baseline: the baseline metric without any of the columns permutated
    scores: differences in baseline metric caused by permutation of each feature, dict in the format {feature:[diffs]}
    '''
    import progressbar
    import random
    # Barra para ver la progresion.
    bar=progressbar.ProgressBar(maxval=len(X.columns))
    # metrica de evaluacion del modelo base.
    baseline_metric=metric(y, clf.predict(X))
    # diccionario por comprension donde tengo keys de variables.
    scores={c:[] for c in X.columns}
    # Inicio la barra de progresion.
    bar.start()
    # Itero sobre cada columna.
    for c in X.columns:
        # Copio el dataframe o vector original.
        X1=X.copy(deep=True)
        # Itero sobre el numero de iteraciones para columna.
        for _ in range(num_iterations):
            # hago una lista temporal.
            temp=X1[c].tolist()
            # Mezclo la columna 
            random.shuffle(temp)
            # Asigno esa lista mezclada a la columna original.
            X1[c]=temp
            # Calculo la metrica seleccionada sobre la columna mezclada.
            score=metric(y, clf.predict(X1))
            # agrego la metrica sobre el el diccionario. 
            scores[c].append(baseline_metric-score)
        # Actualizo la barra de progression.
        bar.update(X.columns.tolist().index(c))
    return baseline_metric, scores

def PermImportanceplot(X, y, clf, metric, num_iterations=100):
    import plotly_express as px
    baseline, scores = PermImportance(X, y, clf, metric, num_iterations)
    percent_changes={c:[] for c in X.columns}
    for c in scores:
        for i in range(len(scores[c])):
            percent_changes[c].append(scores[c][i]/baseline*100)
    
    return(px.bar(pd.DataFrame.from_dict(percent_changes).melt().groupby(['variable']).mean().reset_index().sort_values(['value'], ascending=False)[:25], 
        x='variable', 
        y='value', 
        labels={
            'variable':'column', 
            'value':'% change in recall'
            }))
def score(obj,array):
    return (obj.predict_proba(array)[:,0]*1000).astype(int)

def get_metrics_summary(model,scalerObj,X_test_OOB,y_test_OOB,sc_test_OOB):
    
    scores_df               = pd.DataFrame()
    scores_df['originacion']= sc_test_OOB
    scores_df['incremento'] = score(model,scalerObj.transform(X_test_OOB))
    scores_df['bad']        = y_test_OOB
    scores_df.reset_index(drop=True,inplace=True)
    scores_df['categories_originacion'], edges_o = pd.qcut(scores_df.originacion, 5, retbins=True)
    scores_df['categories_incremento'], edges_i = pd.qcut(scores_df.incremento, 5, retbins=True)
    display(scores_df.groupby('categories_originacion')['bad'].mean()*100)
    display(scores_df.groupby('categories_incremento')['bad'].mean()*100)
    display(get_ks(scores_df[['originacion','bad']],'originacion'))
    display(get_ks(scores_df[['incremento','bad']],'incremento'))
    dual_c=pd.concat([scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].count().unstack().fillna(0),(scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].mean()*100).unstack().fillna(0)],axis=1)
    dual_s=pd.concat([scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].sum().unstack().fillna(0),(scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].mean()*100).unstack().fillna(0)],axis=1)
    display(dual_c)
    display(dual_s)
    print(np.mean(dual_c.iloc[1:,6:].values))

def get_tabla_dual(df):
    scores_df=df.copy()
    scores_df['categories_originacion'], edges_o = pd.qcut(scores_df.score, 5, retbins=True)
    scores_df['categories_incremento'], edges_i = pd.qcut(scores_df.score_incr, 5, retbins=True)
    display(scores_df.groupby('categories_originacion')['bad'].mean()*100)
    display(scores_df.groupby('categories_incremento')['bad'].mean()*100)
    dual_c=pd.concat([scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].count().unstack().fillna(0),(scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].mean()*100).unstack().fillna(0)],axis=1)
    dual_s=pd.concat([scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].sum().unstack().fillna(0),(scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].mean()*100).unstack().fillna(0)],axis=1)
    print('promedio:',np.mean([item for sublist in dual_c.iloc[1:,6:].values for item in sublist]))
    display(dual_c)
    display(dual_s)

def get_mora_ftable(df):
    scores_df=df.copy()
    scores_df['categories_originacion'], edges_o = pd.qcut(scores_df.score, 5, retbins=True)
    scores_df['categories_incremento'], edges_i = pd.qcut(scores_df.score_incr, 5, retbins=True)
    t=scores_df.groupby('categories_originacion')['bad'].mean()*100
    display(t)
    

def get_N_samples_scores(N,df_porcentajes,origen='PADRON'):
    """Obtengo una muestra de tamanio N representado una muestra de PADRON/CONSULTAS de siisa
    Ejemplo: get_N_samples(1000,PADRON,df_porcentajes)
    donde df_porcentajes es un df donde esta la distribucion por scorecard de la poblacion original."""
    for scrd in [3,4,5]:
        if origen=='PADRON':
            origen_sel = df_porcentajes[df_porcentajes.index=='PADRON']
        if origen=='CONSULTAS':
            origen_sel = df_porcentajes[df_porcentajes.index=='CONSULTAS']
        n_sel  = round(float(N*(origen_sel[scrd].values/100)))
        if scrd==3:
            row_sel_3 = df3.sample(n_sel)
            row_sel_3['score_incr']=si.score(model_sc3,scaler_sc3.transform(row_sel_3[vars_sc3]))
            row_sel_3['scorecard']=3
        if scrd==4:
            row_sel_4 = df4.sample(n_sel)
            row_sel_4['score_incr']=si.score(model_sc4,scaler_sc4.transform(row_sel_4[vars_sc4]))
            row_sel_4['scorecard']=4
        if scrd==5:
            row_sel_5 = df5.sample(n_sel)
            row_sel_5['score_incr']=si.score(model_sc5,scaler_sc5.transform(row_sel_5[vars_sc5]))
            row_sel_5['scorecard']=5
    # Agrego los subDfs todos juntos.
    return pd.concat([row_sel_3[['score_incr','score','scorecard','bad']],row_sel_4[['score_incr','score','scorecard','bad']],row_sel_5[['score_incr','score','scorecard','bad']]],axis=0) 
    
def get_GeraldPlot(src='PADRON',rplot=False,niter=100):
    """Funcion que a partir de un df devuelve el grafico de barras ordenado por categoria. """
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    
    badDf=pd.DataFrame()
    for i in range(niter):
        tmp_padron = get_N_samples_scores(50000,porcentajes,'PADRON')
        scores_df=tmp_padron.copy()
        scores_df['categories_originacion'], edges_o = pd.qcut(scores_df.score, 5, retbins=True,labels=['Q1','Q2','Q3','Q4','Q5'])
        scores_df['categories_incremento'], edges_i = pd.qcut(scores_df.score_incr, 5, retbins=True,labels=['Q1','Q2','Q3','Q4','Q5'])
        data = scores_df.groupby(['categories_originacion','categories_incremento'])['bad'].mean().fillna(0).reset_index()
        badDf[i]=data.bad*100
    resultsDF=data[['categories_originacion','categories_incremento']]
    resultsDF['bad_avg']=badDf.mean(axis=1)
    resultsDF['bad_std']=badDf.std(axis=1)
    data= resultsDF.copy()
    labels = data.categories_originacion.unique()
    bars = []
    colors = {'Q1': 'red',
              'Q2': 'orange',
              'Q3': 'lightgreen',
              'Q4': 'forestgreen',
              'Q5': 'darkgreen'}
    for label in labels:
        X     = data[data.categories_incremento==label].categories_originacion.values
        y     = data[data.categories_incremento==label].bad_avg.values
        y_err = data[data.categories_incremento==label].bad_std.values
        
        bars.append(go.Bar(x=X,
                           y=y,
                           error_y=dict(type='data', array=y_err),
                           name=label,
                          marker={'color': colors[label]},))
    f = go.FigureWidget(data=bars)
    f.update_layout(
    title="Model Validation",
    xaxis_title="Risk Score (T=0)",
    yaxis_title="90+ Default Rate",
    font=dict(
        size=18#, color="#7f7f7f"
    )
    )
    if rplot:
        iplot(f, image='svg', filename='barplot', image_width=800, image_height=600);
        f.write_html("file.html")
        f.write_image()
    return f
def get_incremento_tables (df,labels=['bajo','medio','alto'],cocientename='ratio_capital_monto',scorename='score',badname='bad',show_edges=False):
    """Argumentos lista de grupos de incrementos. La lista supone grupos ordenados de menor a mayor en el incremento de deuda
    df es un dataframe que contiene la marca de bad y el score que queremos testear """
    df_current = df.copy()
    # Genero los intervalos de score
    scores,edges = pd.qcut(df_current[scorename],5,retbins=True)
    df_current.loc[:,'bins']=scores.copy()
    # Genero los N grupos de incremento.
    quantiles_cociente, edges_cociente = pd.qcut(df_current[cocientename],len(labels),labels=labels,retbins=True) 
    df_current['quantiles'] = quantiles_cociente.copy()
    if show_edges:
        print(cocientename,edges_cociente)
        print(scorename,edges)
    concat_list = []
    for grupo in labels:
        grupo = df_current[df_current['quantiles']==grupo]
        grupo = grupo.groupby('bins')[badname].mean()*100
        #grupo = grupo.groupby('bins')[badname].count()
        concat_list.append(grupo)
    tabla_incremental = pd.concat(concat_list,axis=1)
    tabla_incremental.columns = labels
    return tabla_incremental