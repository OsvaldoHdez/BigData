# Unit-4 Final Project
---
## Contents
- ### [Introduction]()
- ### [Theoretical framework]()
    - #### [Support Vector Machine]()
    - #### [Decision Tree]()
    - #### [Logistic Regression]()
    - #### [Multilayer Perceptron]()
- ### [Implementation]()
- ### [Results]()
- ### [Conclusions]()
---
## Introduction
The objective of this document is to perform a performance comparison of different classification algorithms, such as Linear Support Vector Machine, Decision Tree, Logistic Regression and Multilayer Perceptron, through apache spark with scala, running the algorithms around thirty times each, in order to make a better performance comparison between each of the machine learning algorithms. 

## Theoretical framework
- ### Support Vector Machine
    Support Vector Machine is one of the classic machine learning techniques that can still help solve big data classification problems. Especially, it can help multi-domain applications in a big data environment. 
    La máquina de vectores de soporte (SVM) es un clasificador lineal binario no probabilístico desarrollado de acuerdo con la minimización del riesgo estructural y el aprendizaje estadístico. Las SVM utilizan un proceso de aprendizaje supervisado para generar funciones de mapeo de entrada-salida a partir de los datos de entrada.

    **Some features:**
    - SVM offers a principles-based approach to machine learning problems due to its mathematical foundation in statistical learning theory.
    - SVM builds its solution in terms of a subset of the training input.
    - SVM has been widely used for classification, regression, novelty detection, and feature reduction tasks.

    #### Functioning
    Una máquina de vectores de soporte construye un hiperplano o un conjunto de hiperplanos en un espacio de dimensión alta o infinita, que se puede utilizar para clasificación, regresión u otras tareas. Intuitivamente, se logra una buena separación por el hiperplano que tiene la mayor distancia a los puntos de datos de entrenamiento más cercanos de cualquier clase (el llamado margen funcional), ya que en general, cuanto mayor es el margen, menor es el error de generalización del clasificador.

- ### Decision Tree
    Decision trees and their sets are popular methods for machine learning regression and classification tasks. Decision trees are widely used because they are easy to interpret, handle categorical features, extend to multiclass classification settings, require no feature scaling, and can capture feature non-linearities and interactions. Tree set algorithms, such as random forests and momentum, are among the best for classification and regression tasks. 

    **Features of a decision tree**
    - The decision tree consists of nodes that form a rooted tree, which means that it is a directed tree with a node called a "root" that has no leading edges.
    - All other nodes have exactly one leading edge. A node with leading edges is called an internal or test node.
    - All other nodes are called leaves (also known as decision or terminal nodes).
    - In a decision tree, each internal node divides the instance space into two or more subspaces according to a certain discrete function of the values of the input attributes. 
    
    #### Functioning
    In the simplest and most common case, each test considers a single attribute, so the instance space is partitioned according to the value of the attribute. In the case of numeric attributes, the condition refers to a range.

    Each sheet is assigned to a class that represents the most appropriate target value. Alternatively, the sheet can contain a probability vector indicating the probability that the target attribute has a certain value. Instances are classified by navigating from the root of the tree to a leaf, according to the results of the tests along the path. 

    <html><div align="center"><img src="https://i.ibb.co/hMRxqzR/Screenshot-2021-06-23-at-10-07-42-Proyecto-final.png"></div></html>

- ### Logistic Regression
    Logistic regression is a statistical instrument for multivariate analysis, of both explanatory and predictive use. Its use is useful when there is a dichotomous dependent variable (an attribute whose absence or presence we have scored with the values zero and one, respectively) and a set of predictive or independent variables, which can be quantitative (which are called covariates or covariates). or categorical. In the latter case, it is required that they be transformed into "dummy" variables, that is, simulated variables. 

    #### Purpose
    The purpose of the analysis is to: predict the probability that a certain “event” will happen to someone: for example, being unemployed = 1 or not being unemployed = 0, being poor = 1 or not poor = 0, receiving a sociologist = 1 or not received = 0).

    Determine which variables weigh more to increase or decrease the probability that the event in question will happen to someone. 

    #### Example
    For example, the logistic regression will take into account the values assumed in a series of variables (age, sex, educational level, position in the home, migratory origin, etc.) the subjects who are effectively unemployed (= 1) and those who they are not (= 0). Based on this, it will predict to each of the subjects - regardless of their real and current state - a certain probability of being unemployed (that is, of having a value of 1 in the dependent variable). 

- ### Multilayer Perceptron
    The multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network.

    The multilayer perceptron is the best known and most widely used type of neural network. In most cases, signals are transmitted within the network in one direction: from input to output. There is no loop, the output of each neuron does not affect the neuron itself. This architecture is called feed-forward and can be seen in the following image.

    <html><div align="center"><img src="https://i.ibb.co/FXvqSZB/Screenshot-2021-06-23-at-10-12-59-Proyecto-final.png"></div></html>

    The power of the multilayer perceptron comes precisely from non-linear activation functions. Almost any nonlinear function can be used for this purpose, except polynomial functions. Currently, the most widely used functions today are the unipolar (or logistic) sigmoid.

    #### Functioning
    MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. The nodes in the input layer represent the input data.
    All other nodes map inputs to outputs by linearly combining the inputs with the node's weights and bias and applying a trigger function.

    #### Some applications
    The multilayer perceptron is an example of an artificial neural network that is widely used for solving a number of different problems, including pattern recognition and interpolation.  

### Implementation

For the implementation, the apache spark tools are used with scala, since Apache Spark is a unified analysis engine for the processing of big data, what we are looking for in the course, with integrated modules for transmission, SQL, machine learning and graphics processing.

By having the aforementioned characteristics, it allows us to carry out everything proposed in this document. In order to have a good analysis of the aforementioned learning algorithms and make a comparison between them.

You will work with the scala programming language because scala combines functional and object-oriented programming in a concise high-level language.

Scala's static types help avoid errors in complex applications, and its JVM and JavaScript runtimes allow you to build high-performance systems with easy access to huge library ecosystems.

As mentioned above, scala helps JVM (Java Virtual Machine) runtimes, this is useful to us since we will make the comparison between different learning algorithms and scala helps us to execute them in a better way.

The implementation is done on a desktop computer with a Linux base operating system, in the source code editor "Visual Code".

