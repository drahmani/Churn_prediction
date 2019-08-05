# Telco Customet Churn prediction
Churn (attrition) in the context of a subscriber-based service model is the measure of the
number of individuals that leave the supplier during a given time period. Churn rate is a
possible indicator of customer dissatisfaction, cheaper and/or better offers from the
competition, more successful sales and/or marketing by the competition, or reasons having
to do with the customer life cycle.

**Aim**: Predicting churn is a very important factor for service providers that allows for taking
various actions that prevent churn and avoid revenue loss.


_**Overview of my code**_

Note: code is in churn_1.ipynb or gist https://gist.github.com/drahmani/07ce109f49195ff06e1cd4eac94c2d78

1. **Data loading and preprocessing**

  	1. filled in NaN’s
  	2. examined histograms
  	3. inferred 11 missing values for total charges that could be re-calculated from data.

2. **Data encoding**

	1. Converted target variable to 1-hot encoding → [0,1]
	2. Build numeric feature columns
	3. Build categorical feature columns – 1-hot encoding (there will be no collisions in the hash 		 
            bucket as the hash unique categories is quite small).
	4. Split data into test and train sets
	5. Create input functions which will be passed to the estimators.

3. **Linear classifier (logistic regression)**
	The result looks ok for a first attempt, second class ('Yes') is not being classified well [242 192] as the data is unbalanced (many more category 0 than 1 – will do balancing later on). This is the baseline. C=[1218 109][242 192]

4. **DNN classifier.**
	First needed to figure out a rough size for the network layers. We have 19 features, 2 classes so a first guess would be a simple network. I tried the following combinations but omit the details for brevity. (regularization will be attempted later).
 	1. 10x4 [layer 1 x layer2]. Confusion matrix row for class 2 is awful.
 	2. 4x10. Attempt a choke point in first layer – problem remains.
 	3. 30x10. Attempt a more complex network perhaps the network was underfitting – problem remains.
 	4. 64x32. This is really as complex as the network should be allowed. Worse again – 			the network is overfitting now.
 	5. 4x2. Everyone classified in class 1 – network is unable to draw a plane of 				separability.
 	6. 4x4. Much better.
 	7. 6x4. Better again – here is roughly where the complexity matches the problem.
 	8. 8x6. C = [984 343][80 354]

5. **Introduce regularisation.**
	1. Linear model with L1 (abs value of the parameters – encourages sparse parameter set as will drive some to 0) & L2 regularisation (squared parameters). Slight improvement. C = [1142 185][196 238]

	2. DNN with regularisation. Note: sometimes over-regularising can cause the prediction accuracy to degrade if the regularization weights are too high. Results seem to be worse, weakening the L2 term. C= [1153 174][186 248]

6. **Feature engineering.** Created
	1) cross term of [onlinesecurity, online backup, and device protection.]
	2) Bucketised tenure into intervals [6,12,18,24,36,48,60,80]

Linear classifier (no regularisation) result: C = [1073 254][133 301]
        1. Trying buckets of 6 months, 1 year, 2 years, 2+ i.e. [6,12,24]
	2. Linear classifier (no regularisation) result: C = [1076 251][139 295]  - more or less the same.
	3. Linear classifier (+ regularisation) result: C = [1013 314][112 322] – much better for class 2. compare to C = [1142 185][196 238] without buckets => better for class 2 but we lose on class 1 but its preferable as its more balanced.

 
7. **Balancing data set.**
	It's likely the end result of this model are customer calls made to people likely to churn. Calling someone unlikely to churn causes a smaller cost than missing someone that actually churns => the cost of two types of errors are not equal. Errors in class 2 cost far more than in class 1 => bias the number of samples to reflect this.
 	
	**Rebalanced the data using ROS(RandomOverSampler)** resulting in a new resampled data set with [3847 3847] examples from each class. Note the test set is left untouched.

	Linear (no reg) C = [375 952][27 407] => we will catch most of the people likely to churn ut 	ring ~1000 who won’t – not good result.

	Linear (+ reg) C = [1177 150][200 234] => we will miss ½  the people likely to churn – not 	good result.

	DNN (+reg) C =[1163 163][186 248] => Not that different from the unbalanced data set! 	Not a good result we will miss ~2/5 of the churners.

8. **Kernel model estimates**
	Given that initial analysis showing that neither the linear nor the DNN show significantly different results lets try a model which places a medium level of assumptions on the data (a linear model places the maximum number of assumptions, the DNN places none).  A kernel model still assumes linearity but in a higher-dimensional space.

	Note: code for kernel estimator is in churn_2.ipynb

	The model uses rebalancing & regularisation.
	C = [915 412][74 360]
	This is a much better result – we identify most of the churners but only 400 non-churners would be called
	
9. **Results summary**
	Below we show three graphs comparing the accuracy, F1 Score and specifically the number of false positives for class 1 (churners) as this is the only class we care about. The aim is to achieve a high true positive for class 1 balanced against the desire to have a low false positive.

Based on the assumption that we will be using the classifier to make customer calls the DNN without regularisation or balancing appears to give the best trade-off in the models tested. The kernel model gives a slightly better identification of the true positives but with a corresponding increase in false positives. With more investigation, the DNN could be tuned appropriately with regularisation and balancing to obtain a better result (the models attempted probably had a regularisation parameter set too high).

Alternative models such as an SVM, boosted gradient trees and ensembles (averaging over several model predictions) were not tested but are the next directions for investigation.

![acc](https://user-images.githubusercontent.com/38282927/62459476-5d8ca380-b777-11e9-9af7-978b6563c537.png)

   Figure 1. Accuracy



![f1](https://user-images.githubusercontent.com/38282927/62459477-5d8ca380-b777-11e9-92ab-5b8579d75ece.png)
    
   Figure 2. F1 Score.


![false_pos](https://user-images.githubusercontent.com/38282927/62459478-5d8ca380-b777-11e9-9570-44e0608d2fb8.png)

   Figure 3. False positive and true positives for Class 1 (churners).


