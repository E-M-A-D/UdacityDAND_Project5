#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


###############################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus', 'total_stock_value', 'salary', \
                 'exercised_stock_options', 'total_payments', 'lti_ratio']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
temp_data_dict = data_dict.copy()

#lti_ratio is the ratio of long term incentives and the total payments. The higher
#the number, the more the person opted to invest in Enron stocks
for key, value in data_dict.iteritems():
    if (temp_data_dict[key]['long_term_incentive'] == 'NaN') or \
       (temp_data_dict[key]['total_payments'] == 'NaN'):
        lti_val = 'NaN'
    else:
        lti_val = float(temp_data_dict[key]['long_term_incentive'])/temp_data_dict[key]['total_payments']

    temp_data_dict[key]['lti_ratio'] = lti_val
    
    for param in value.keys():
        if param not in features_list:
            temp_data_dict[key].pop(param)
            

###############################################################################
### Task 2: Remove outliers
temp_data_dict.pop('TOTAL')
temp_data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
temp_data_dict.pop('LOCKHART EUGENE E')



###############################################################################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = temp_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

from sklearn.preprocessing import Imputer, MinMaxScaler

labels, features = targetFeatureSplit(data)
#These were included anyway. I know that for my final choice, a decision tree,
#feature scaling doesn't really matter, but this part was used when
#I was trying out other classifiers. And since it doesn't hurt neither the performance
#of a decision tree, it's not a big deal to keep it.
min_max_scaler = MinMaxScaler()
imputer = Imputer(strategy="mean")
features = imputer.fit_transform(features)
features = min_max_scaler.fit_transform(features)
#print f[0]
###############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation  import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn import svm

clf = DecisionTreeClassifier(max_depth = 5,
                             max_features =  None,
                             class_weight = {0:1, 1:2} )


###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

X_train, X_test, y_train, y_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

skf = StratifiedKFold(y_train, n_folds = 7)
target_names = ['NON POI', 'POI']

for train_index, validation_index in skf:

    current_training_set = [ X_train[i] for i in train_index]
    current_training_labels = [ y_train[i] for i in train_index]
    clf.fit( current_training_set, current_training_labels)

    current_testing_set = [ X_train[i] for i in validation_index]
    current_testing_labels = [ y_train[i] for i in validation_index]
    
    prediction_result = clf.predict(current_testing_set)
    print classification_report(current_testing_labels,
                                prediction_result,
                                target_names=target_names)

    cm = confusion_matrix(current_testing_labels, prediction_result)
    print "CONFUSION MATRIX:\n", cm
    print "----------------------------------------------------------------------------"

test_prediction_result = clf.predict(X_test)

cm = confusion_matrix(y_test, test_prediction_result)
print "CONFUSION MATRIX FOR PREDICTION:\n", cm

#Tuning
#This part was done in the Jupyter Notebook, I have put here the commented code just for a reference
#decision_tree_grid_param = {'criterion' : ['gini'], #Discard entropy, since these are just numerical data, not classes.
#                            'max_depth' : [None], 
#                            'max_features' : [None, 'sqrt', 'log2'] ,
#                            'presort' : [True], #this is a small dataset, so let's speed things up,
#                            'class_weight': [{0: 1, 1: poi_weight} for poi_weight in range(1,10)]
#                           }
#decision_tree_estimator = DecisionTreeClassifier()
#decision_tree_grid_clf = grid_search.GridSearchCV(decision_tree_estimator, 
#                                                  decision_tree_grid_param, cv=10)
#print "Best Score: ", decision_tree_grid_clf.best_score_
#print "Best Parameters: ", decision_tree_grid_clf.best_params_


#Tuning for the tester.py
#As my target was better scores over the tester.py, I have used this commented code to tune my algorithm.
#It is a hand-written GridSearchCV that calls the tester.py after fitting the model.
#import tester
#for i in range(0,9):
#    for j in range(0,7):
#        if i == 0:
#            md = None
#        else:
#            md = i
#
#        if j == 0:
#            mf = None
#        else:
#            mf = j
#
#        for k in range(1,9):
#            
#            clf = DecisionTreeClassifier(max_depth = md,
#                                     max_features =  mf,
#                                     class_weight = {0:1, 1:k} )
#            clf.fit(X_train, y_train)
#            #test_prediction_result = clf.predict(X_test)
#            dump_classifier_and_data(clf, my_dataset, features_list)
#            tester.main()


###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
