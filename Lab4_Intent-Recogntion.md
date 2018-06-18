---
title: "Tutorial 4: Intent Recognition"
author:
- 'Younesse Kaddar'
- 'Alexandre Olech'
- 'Kexin Ren'

date: 2018-06-05
tags:
  - lab
  - tutorial
  - exercise

abstract: 'Lab 4: Intent Recognition'
---

# Lab 4: Intent Recognition
### Younesse Kaddar, Alexandre Olech and Kexin Ren (**Lecturers**: ???)

# Exercice 1. Automatic detection of speaker’s intention from supra-segmental feat

In this exercice, we consider a Human-Robot Interaction situation in which a Human is evaluating actions performed by the Kismet robot: approval or prohibition. The initial corpus contains a total of 1002 American English utterances of varying linguistic content produced by three female speakers in five classes of affective communicative intents. The classes are Approval, Attention, Prohibition Weak, Soothing, and Neutral utterances. The affective intents sound acted and are generally expressed rather strongly. The speech recordings are of variable length, mostly in the range of 1.8 - 3.25s. We extracted prosodic features such as $f_0$ and $energy$. Files are respectively named *.$f0$ and *.$en$ (time, value)


# Steps


## 1. Extraction of prosodic features ($f_0$ and $energy$)

We found that, in the given data, the files are labelled as "$at$", "$pw$" or "$ap$", which could represent the intention classes "Attention", "Prohibition Weak" and "Approval" respectively. Since the aim of this exercice is to develop a human feedback classifier for positive ("Approval") / negative ("Prohibition") intentions, we extract the files labelled as "$pw$" and "$ap$" and exclude the files labelled as "$at$".

Our project was cooperated online using Google Colab. The codes for extracting the files labelled as "$pw$" and "$ap$" and extracting the prosodic features shown are as below:

```python
 filenames = list_from_URL('https://raw.githubusercontent.com/youqad/Neurorobotics_Intent-Recognition/master/filenames.txt')
 filenames = list(set(filenames))
 
 files = []
 indices = []
 
 for file in filenames:
 
     URL_f0 = 'https://raw.githubusercontent.com/youqad/Neurorobotics_Intent-Recognition/master/data_files/{}.f0'.format(file)
     file_dicts = [{key:val for key, val in zip(['time', 'f0'], map(float, l.split()))} for l in list_from_URL(URL_f0)]
 
     URL_en = 'https://raw.githubusercontent.com/youqad/Neurorobotics_Intent-Recognition/master/data_files/{}.en'.format(file)
     for l, d in zip(list_from_URL(URL_en), file_dicts):
       d["file"] = file
       d["en"] = float(l.split()[1])
       d["label"] = file[-2:]
 
     files.extend(file_dicts)
 
# How `files` looks like:
# # files = [ 
# #           {"file": "cy0001at", "time": 0.02, "f0": 0., "en": 0.},
# #           {"file": "cy0001at", "time": 1.28, "f0": 0., "en": 0.},
# #           ...
# #           {"file": "li1450at", "time": 0.02, "f0": 0., "en": 0.},
# #           {"file": "li1450at", "time": 1.56, "f0": 404., "en": 65.}
# #         ]
 
 pd.DataFrame(files).to_csv('data.csv', encoding='utf-8', index=False) # To reuse it next time
 google_files.download('data.csv')
 
 # loading training data
 df = pd.read_csv('https://raw.githubusercontent.com/youqad/Neurorobotics_Intent-Recognition/master/data.csv').set_index('file')
 df1 = df.loc[df['label'] != 'at']
```


## 2. Extraction of functionals (statistics) : mean, maximum, range, variance, median, first quartile, third quartile, mean absolute of local derivate

We calculated the mean, max, range, variance, median. first quartile, third quartile and mean absolute pf local derivate for each $en$ and $f0$ file. The codes for the extraction of the functionals above are as below:

```python
list_features  = ['mean', 
                  'max',
                  ('range', lambda x: max(x)-min(x)),
                  'var',
                  'median',
                  ('1st_quantile', lambda x: x.quantile(.25)),
                  ('3rd_quantile', lambda x: x.quantile(.75)),
                  ('mean_absolute_local_derivate', lambda x: abs(x.diff()).mean())
                 ]

df1.groupby('file')['f0','en'].agg(list_features).head()
```

Table $1$ and Table $2$ show the first five lines of the statistics of $f0$ and $en$ files respectively:

### TABLE $1$ Statistics of $f0$ files (first $5$ lines)
	
|file|mean|	max|	range|	var|	median|	$1$st_quantile|	$3$rd_quantile|	mean_absolute_local_derivate|																	
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|	92.3	|257.0|	257.0|	10372.5|	0.0|	0.0|	189.5|	13.7|	
|cy0008pw	|78.4	|250.0|	250.0|	9930.1|	0.0|	0.0|	192.0|	26.4|	
|cy0009pw	|69.1	|243.0|	243.0|	8927.2|	0.0|	0.0|	182.3|	12.9|	
|cy0010pw	|29.2	|221.0|	221.0|	4696.2|	0.0|	0.0|	0.0|	15.27|	
|cy0011pw	|110.7	|230.0|	230.0|	9290.4|	172.0|	0.0|	192.5|	7.5|	


### TABLE $2$ Statistics of $en$ files (first $5$ lines)

|file|mean|	max|	range|	var|	median|	$1$st_quantile|	$3$rd_quantile|	mean_absolute_local_derivate|
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|52.3|	71.0|	71.0|	228.5|	52.0|	41.0|	66.0|	2.9|
|cy0008pw	|47.7|	70.0|	70.0|	321.9|	43.0|	41.0|	64.5|	3.9|
|cy0009pw	|49.5|	74.0|	74.0|	260.8|	42.0|	40.8|	66.0|	3.5|
|cy0010pw	|46.1|	77.0|	77.0|	165.8|	42.0|	41.0|	50.8|	3.3|
|cy0011pw	|53.7|	71.0|	71.0|	258.1|	62.0|	41.3|	66.0|	2.3|


## 3. Check functionals for both voiced (i.e. $f_0$ ≠ $0$) and unvoiced segments. Which segments are suited for the approach?

We extract voiced segments by looking for the data whose $f_0$ value is not equal to $0$. The codes for extracting voiced sections and calculating the statistics of them are:

```python
voiced = df1.loc[df1['f0']!=0].groupby('file')['f0','en'].agg(list_features)
voiced.head()
```

Similarly, we extract the unvoiced segments by looking for the data whose $f_0$ value equals to $0$. The codes for extracting unvoiced segments and calculating the statistics of them are:

```python
unvoiced = df1.loc[df1['f0']==0].groupby('file')['en'].agg(list_features)
unvoiced.head()
```

The first $5$ lines of statistics for voiced segments in $f0$ and $en$ files are shown in Table $3$ and Table $4$ respectively:

### TABLE $3$ Statistics of $f0$ files of voiced segments (first $5$ lines)
	
|file|mean|	max|	range|	var|	median|	$1$st_quantile|	$3$rd_quantile|	mean_absolute_local_derivate|		
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|	200.3|	257.0|	90.0|	675.9|	191.0|	182.5|	213.0|	5.9|	
|cy0008pw|	200.0|	250.0|	83.0|	538.4|	198.5|	179.5|	210.0|	10.4|	
|cy0009pw|	194.4|	243.0|	77.0|	446.9|	190.0|	180.0|	209.0|	7.2|	
|cy0010pw|	186.1|	221.0|	67.0|	465.3|	178.5|	171.3|	204.3|	6.5|	
|cy0011pw|	191.9|	230.0|	66.0|	314.8|	190.0|	179.0|	204.0|	4.1|	

### TABLE $4$ Statistics of $en$ files of voiced segments (first $5$ lines)

|file|mean|	max|	range|	var|	median|	$1$st_quantile|	$3$rd_quantile|	mean_absolute_local_derivate|
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|65.9|	71.0|	16.0|	17.8|	66.0|	63.5|	70.0|	1.7|
|cy0008pw	|61.0|	70.0|	70.0|	242.7|	66.0|	61.5|	68.0|	5.8|
|cy0009pw	|67.3|	74.0|	20.0|	17.9|	68.0|	66.0|	70.0|	2.9|
|cy0010pw	|65.8|	77.0|	25.0|	50.5|	64.0|	62.0|	70.8|	4.0|
|cy0011pw	|65.3|	71.0|	19.0|	14.7|	65.0|	63.0|	68.0|	0.9|

The first $5$ lines of statistics of unvoiced segments for $en$ files are shown in Table $5$:

### TABLE $5$ Statistics of $en$ files of unvoiced segments (first $5$ lines)

|file|mean|	max|	range|	var|	median|	$1$st_quantile|	$3$rd_quantile|	mean_absolute_local_derivate|
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |								
|cy0007pw|	40.7|	58.0|	58.0|	113.6|	41.0|	40.5|	43.5|	3.7|
|cy0008pw|	39.2|58.0|	58.0|	189.6|	42.0|	41.0|	43.0|	5.2|
|cy0009pw|	39.6|	56.0|	56.0|	119.6|	41.0|	40.0|	42.0|	3.6|
|cy0010pw|	42.4|	68.0|	68.0|	101.4|	41.0|	40.0|	43.0|	3.1|
|cy0011pw|	37.8|	51.0|	51.0|	150.9|	41.0|	40.0|	42.0|	4.1|

To judge which segment is better for the approach, we should check how similar the data is wthin the same group and how different the data is in different groups. We firstly look at the overall statistics of the two segments for the two classes "Approval" and "Prohibition Weak". The results are shown in Table $6$:

### TABLE $6$ Statistics of "Approval" ($ap$) and "Prohibition Weak" ($pw$) files for voiced and unvoiced segments

|segments|file|class|mean|	max|	range|	var|	median|	$1$st_quantile|	$3$rd_quantile|	mean_absolute_local_derivate|
| --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |	--- |	
|voiced|f0| ap| 289.5|	597.0|	521.0|	11013.0|272.0|	199.0|	370.5|	24.9|	
|voiced|f0| pw|192.4|	597.0|	522.0|	2702.1|	191.0|	170.0|	218.0|	14.4|
|voiced|en|ap|73.3|	93.0|	93.0	|88.2|	74.0|	68.0|	79.0|	3.5|
|voiced|en|pw|71.6|	91.0|	91.0|	84.7|	72.0|	65.0|	79.0|	3.0|	
|unvoiced|f0| ap|0.0|	0.0|	0.0|	0.0|	0.0|	0.0|	0.0|	0.0|
|unvoiced|f0| pw|0.0|	0.0|	0.0|	0.0|	0.0|	0.0|	0.0|	0.0|
|unvoiced|en| ap|46.4|	94.0|	94.0|	239.2|	43.0|	41.0|	55.0|	3.9|
|unvoiced|en| pw|47.6|	91.0|	91.0|	231.1|	47.0|	40.0|	58.0|	3.5|

Table $6$ shows that, in voiced segments, the statistics of $f_0$ are more different for $ap$ and $pw$ files. The $en$ files of voiced and unvoiced segments are very close between the two different class ($ap$ and $pw$), and of course, the statistics $f0$ files of unvoiced segments are all 0. Thus, so far from the results of Table 6, $f_0$ values of voiced segments seem a better measurement for the approach. 

We will work on this problem more in Question 1.4, by plotting the data of different classes and observing their similarity in same group and separability in different groups.

## 4. Build two databases by randomly extracting examples : Learning database ($60$%) and Test database

We randomly extracted $60$% data from original dataset to build the training set and the remaiaining data were used as test set. The codes for this procedure are provided as below:

```python
def train_test(df=df1, train_percentage=.6, seed=1):
  
  voiced = df.loc[df['f0']!=0].groupby('file')['f0','en'].agg(list_features)
  unvoiced = df.loc[df['f0']==0].groupby('file')['en'].agg(list_features)

  X, Y = {}, {}

  X['voiced'], Y['voiced'] = {}, {}
  X['unvoiced'], Y['unvoiced'] = {}, {}


  X['voiced']['all'] = np.array(df.groupby('file')['f0','en'].agg(list_features))
  Y['voiced']['all'] = np.array(df.loc[df['f0']!=0].groupby(['file']).min().label.values)

  X['unvoiced']['all'] = np.array(unvoiced)
  Y['unvoiced']['all'] = np.array(df.loc[df['f0']==0].groupby(['file']).min().label.values)
  
  np.random.seed(seed)
  
  for type in ['voiced', 'unvoiced']:
    n = len(X[type]['all'])
    ind_rand = np.random.randint(n, size=int(train_percentage*n)) # random indices
    train_mask = np.zeros(n, dtype=bool)
    train_mask[ind_rand] = True
    X[type]['train'], X[type]['test'] = X[type]['all'][train_mask],  X[type]['all'][~train_mask]
    Y[type]['train'], Y[type]['test'] = Y[type]['all'][train_mask],  Y[type]['all'][~train_mask]
  
  return X, Y

X1, Y1 = train_test()
```

Remind that in Question $1.4$, we found that the $f_0$ values of voiced segments might be the better measurement for classifying. Thus, we plot the training data of voiced segments in the $2$d coordinate system consitituted by $variance$ and $mean absolute local derivate$ of $f0$ files. The plot is shown as Figure 1:


<img src="https://github.com/youqad/Neurorobotics_Intent-Recognition/blob/master/fig1.png" alt=" Variance and mean absolute local derivate of voiced segments" style="margin-left: 7%;"/>
### FIGURE $1$ Variance and mean absolute local derivate of $f_0$ in voiced segments


Figure 1 indicates that the Approval data nad Prohibition Weak data can be separated well using their variance and mean absolute local derivate of $f_0$ values in voiced segments, supporting our idea in Question 1.3 that $f_0$ values of voiced segments can be a good measurement for classifying.


We also plot the data using variance and mean absolute local derivate of $energy$ values in unvoiced segments. The plot is shown in Figure 2:


<img src="https://github.com/youqad/Neurorobotics_Intent-Recognition/blob/master/fig2.png" alt=" Variance and mean absolute local derivate of unvoiced segments" style="margin-left: 7%;"/>
### FIGURE $2$ Variance and mean absolute local derivate of $energy$ inunvoiced segments


Plot 2 shows that the data can not be separated well according to their classes using the variance and mean absolute local derivate of $energy$ inunvoiced segments. 

In sum, by plotting our randomly selected training data, we found that $f_0$ values of voiced segments are the best measurement for classifying.



## 5. Train a classifer (k-NN method)

## 6. Evaluate and discuss the performance of the classifier. You will discuss the relevance of the parameters (f0 et energy), the role of the functionals, the role of k, ratio of Learning/Test databases, random design of databases.


# Exercice 2. Detection of multiple intents :

# Steps

## 1. Extract the prosodic features (f0 and energy) and their functionals
## 2. Develop a classifier for these three classses
## 3. Evaluate and discuss the performance of the classifier. We could use confusion matrices.
