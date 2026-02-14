a. Problem Statement

Equity pledge financing is widely used by controlling shareholders of listed companies to obtain loans by pledging their shares as collateral. Although this financing method helps meet short-term capital needs, it introduces significant credit risk. A decline in stock prices or poor company performance may lead to default, causing financial losses to lenders and instability in capital markets.

The objective of this project is to build and compare multiple machine learning classification models to predict whether an equity pledge financing case will default (1) or remain normal (0). The goal is to identify the most effective model for default risk prediction.

b. Dataset Description

The dataset contains information about equity pledge financing cases of controlling shareholders of Chinese listed companies from 2017 to 2022.

Target Variable: IsDefault (renamed as target in the model)

1 → Default

0 → Normal (No default)

This is a binary classification problem.

The dataset includes various financial and market-related features that may influence the likelihood of default. These features help capture company financial health, market conditions, and risk exposure.

This problem is a supervised learning classification task in the financial risk management domain.

c. Models Used and Performance Comparison

The following six machine learning models were implemented and evaluated:

1. Logistic Regression

2. Decision Tree

3. K-Nearest Neighbors (kNN)

4. Naive Bayes

5. Random Forest (Ensemble)

6. XGBoost (Ensemble)


Model Comparison Table

ML Model	        Accuracy	AUC	    Precision	 Recall	  F1 Score	  MCC

Logistic Regression	0.8651	   0.9436	0.5946	     0.8302	   0.6929	0.6234

Decision Tree	    0.8616	   0.9099	0.5802	     0.8868	   0.7015	0.6400

kNN	                0.8685	   0.7957	0.9412	     0.3019	   0.4571   0.4895

Naive Bayes	        0.8235	   0.8682	0.5147	     0.6604	   0.5785	0.4749

Random Forest   	0.9239	   0.9636	0.8163	     0.7547	   0.7843	0.7390

XGBoost (Ensemble)	0.9896	   1.0000	1.0000	     0.9434	   0.9709	0.9652


Observation about model performance

Logistic Regression : Provided stable baseline performance with good AUC and balanced recall, but moderate precision.

Decision Tree : Showed strong recall but slightly lower precision; may be prone to overfitting.

kNN : Achieved very high precision but extremely low recall, indicating poor balance in detecting positive cases.

Naive Bayes : Performed the weakest overall with lower accuracy, F1 score, and MCC compared to other models.

Random Forest (Ensemble) : Demonstrated strong performance with high accuracy and well-balanced precision and recall.

XGBoost (Ensemble) : Achieved the best overall performance across all metrics, including perfect AUC (1.0000), highest F1 score (0.9709), and highest MCC (0.9652).

