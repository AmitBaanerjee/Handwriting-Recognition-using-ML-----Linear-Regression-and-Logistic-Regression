# Handwriting-matching using Logistic Regression and Linear Regression


1 Overview


You have been hired by the FBI to develop predictive models for detection of crime, your task is to
help the Bureau and police departments to solve criminal cases dealing with evidence provided by
handwritten documents such as wills and ransom notes. You are assigned to a forensic project by the
FBI. The project requires you to apply machine learning to solve the handwriting comparison task in
forensics. We formulate this as a problem of linear regression where we map a set of input features x
to a real-valued scalar target y(x,w).
Your task is to find similarity between the handwritten samples of the known and the questioned
writer by using linear regression.
Each instance in the CEDAR “AND” training data consists of set of input features for each handwritten
“AND” sample. The features are obtained from two different sources:
1. Human Observed features: Features entered by human document examiners manually
2. GSC features: Features extracted using Gradient Structural Concavity (GSC) algorithm.
The target values are scalars that can take two values {1:same writer, 0:different writers}. Although
the training target values are discrete we use linear regression to obtain real values which is more useful
for finding similarity (avoids collision into only two possible values).
