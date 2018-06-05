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


The aim of this exercice is to develop a human feedback classifier : positive (approval)/negative (prohibition). This classifier might be used to teach robots and/or to guide robot’s learning. Development of human feedback classifier :
1. Extraction of prosodic features (f0 and energy)
2. Extraction of functionals (statistics) : mean, maximum, range, variance, median, first quartile, third quartile, mean absolute of local derivate
3. Check functionals for both voiced (i.e. f0 6= 0) and unvoiced segments. Which segments are suited for the approach ?
4. Build two databases by randomly extracting examples : Learning database (60%) and Test database
5. Train a classifer (k-NN method)
6. Evaluate and discuss the performance of the classifier


# Steps


## 1. Extraction of prosodic features (f0 and energy)

...


## 2. Extraction of functionals (statistics) : mean, maximum, range, variance, median, first quartile, third quartile, mean absolute of local derivate

...

## 3. Check functionals for both voiced (i.e. f0 6= 0) and unvoiced segments. Which segments are suited for the approach?

...

## 4. Build two databases by randomly extracting examples : Learning database (60%) and Test database
## 5. Train a classifer (k-NN method)
## 6. Evaluate and discuss the performance of the classifier
