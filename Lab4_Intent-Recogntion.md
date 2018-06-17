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

In this exercice, we consider a Human-Robot Interaction situation in which a Human is evaluating actions performed by the Kismet robot: approval or prohibition. The initial corpus contains a total of 1002 American English utterances of varying linguistic content produced by three female speakers in five classes of affective communicative intents. The classes are Approval, Attention, Prohibition Weak, Soothing, and Neutral utterances. The affective intents sound acted and are generally expressed rather strongly. The speech recordings are of variable length, mostly in the range of 1.8 - 3.25s. We extracted prosodic features such as f0 and energy. Files are respectively named *.f0 and *.en (time, value)


# Steps


## 1. Extraction of prosodic features (f0 and energy)

...


## 2. Extraction of functionals (statistics) : mean, maximum, range, variance, median, first quartile, third quartile, mean absolute of local derivate

The codes for the extraction of the functionals above are as below:
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

The results we obatained for $f0$ and $en$ files are shown in Table $1$ and Table $2$ respectively:

### TABLE $1$ Statistics of $f0$ Files
	
|file|mean|	max|	range|	var|	median|	1st_quantile|	3rd_quantile|	mean_absolute_local_derivate|																	
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|	92.3	|257.0|	257.0|	10372.5|	0.0|	0.0|	189.5|	13.7|	
|cy0008pw	|78.4	|250.0|	250.0|	9930.1|	0.0|	0.0|	192.0|	26.4|	
|cy0009pw	|69.1	|243.0|	243.0|	8927.2|	0.0|	0.0|	182.3|	12.9|	
|cy0010pw	|29.2	|221.0|	221.0|	4696.2|	0.0|	0.0|	0.0|	15.27|	
|cy0011pw	|110.7	|230.0|	230.0|	9290.4|	172.0|	0.0|	192.5|	7.5|	


### TABLE $2$ Statistics of $en$ Files

|file|mean|	max|	range|	var|	median|	1st_quantile|	3rd_quantile|	mean_absolute_local_derivate|
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|52.3|	71.0|	71.0|	228.5|	52.0|	41.0|	66.0|	2.9|
|cy0008pw	|47.7|	70.0|	70.0|	321.9|	43.0|	41.0|	64.5|	3.9|
|cy0009pw	|49.5|	74.0|	74.0|	260.8|	42.0|	40.8|	66.0|	3.5|
|cy0010pw	|46.1|	77.0|	77.0|	165.8|	42.0|	41.0|	50.8|	3.3|
|cy0011pw	|53.7|	71.0|	71.0|	258.1|	62.0|	41.3|	66.0|	2.3|

Table $1$ demonstrates that... (question: stats for each sample file? or for all files?)

Table $2$ demonstrates that...


## 3. Check functionals for both voiced (i.e. $f0$ ≠ $0$) and unvoiced segments. Which segments are suited for the approach?

The codes for calculating the statistics of voiced (i.e. $f0$ ≠ $0$) segments are:

```python
voiced = df1.loc[df1['f0']!=0].groupby('file')['f0','en'].agg(list_features)
voiced.head()
```

The codes for calculating the statistics of unvoiced (i.e. $f0$ = $0$) segments are:

```python
unvoiced = df1.loc[df1['f0']==0].groupby('file')['en'].agg(list_features)
unvoiced.head()
```

The statistics of $f0$ files and $en$ files for voiced segments are shown in Table $3$ and Table $4$ respectively:

### TABLE $3$ Statistics of $f0$ Files of Voiced Segments
	
|file|mean|	max|	range|	var|	median|	1st_quantile|	3rd_quantile|	mean_absolute_local_derivate|		
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|	200.3|	257.0|	90.0|	675.9|	191.0|	182.5|	213.0|	5.9|	
|cy0008pw|	200.0|	250.0|	83.0|	538.4|	198.5|	179.5|	210.0|	10.4|	
|cy0009pw|	194.4|	243.0|	77.0|	446.9|	190.0|	180.0|	209.0|	7.2|	
|cy0010pw|	186.1|	221.0|	67.0|	465.3|	178.5|	171.3|	204.3|	6.5|	
|cy0011pw|	191.9|	230.0|	66.0|	314.8|	190.0|	179.0|	204.0|	4.1|	

### TABLE $4$ Statistics of $en$ Files of Voiced Segments

|file|mean|	max|	range|	var|	median|	1st_quantile|	3rd_quantile|	mean_absolute_local_derivate|
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |
|cy0007pw|65.9|	71.0|	16.0|	17.8|	66.0|	63.5|	70.0|	1.7|
|cy0008pw	|61.0|	70.0|	70.0|	242.7|	66.0|	61.5|	68.0|	5.8|
|cy0009pw	|67.3|	74.0|	20.0|	17.9|	68.0|	66.0|	70.0|	2.9|
|cy0010pw	|65.8|	77.0|	25.0|	50.5|	64.0|	62.0|	70.8|	4.0|
|cy0011pw	|65.3|	71.0|	19.0|	14.7|	65.0|	63.0|	68.0|	0.9|

The statistics for $en$ files for unvoiced segments are shown in Table $5$:

### TABLE $5$ Statistics of $en$ Files of Unvoiced Segments

|file|mean|	max|	range|	var|	median|	1st_quantile|	3rd_quantile|	mean_absolute_local_derivate|
| ---------- | --- |--- |--- |--- |--- |--- |--- |--- |								
|cy0007pw|	40.7|	58.0|	58.0|	113.6|	41.0|	40.5|	43.5|	3.7|
|cy0008pw|	39.2|58.0|	58.0|	189.6|	42.0|	41.0|	43.0|	5.2|
|cy0009pw|	39.6|	56.0|	56.0|	119.6|	41.0|	40.0|	42.0|	3.6|
|cy0010pw|	42.4|	68.0|	68.0|	101.4|	41.0|	40.0|	43.0|	3.1|
|cy0011pw|	37.8|	51.0|	51.0|	150.9|	41.0|	40.0|	42.0|	4.1|

By comparing Table $3$, $4$ and $5$, we find that ...

## 4. Build two databases by randomly extracting examples : Learning database (60%) and Test database

## 5. Train a classifer (k-NN method)

## 6. Evaluate and discuss the performance of the classifier. You will discuss the relevance of the parameters (f0 et energy), the role of the functionals, the role of k, ratio of Learning/Test databases, random design of databases.


# Exercice 2. Detection of multiple intents :

# Steps

## 1. Extract the prosodic features (f0 and energy) and their functionals
## 2. Develop a classifier for these three classses
## 3. Evaluate and discuss the performance of the classifier. We could use confusion matrices.
