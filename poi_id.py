#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                   #'salary',
                   #'total_payments',
                   'bonus',
                   'total_stock_value', 
                   #'expenses',
                   'exercised_stock_options',
                   #'other', 
                   #'long_term_incentive',
                   #'restricted_stock',
                   #'to_messages', 
                   #'from_poi_to_this_person', 
                   #'from_messages',
                   #'from_this_person_to_poi',
                   #'shared_receipt_with_poi',
                   #'total_messages'
                  ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print len(data_dict.items())
## all the data_dict contain 146 items in the dataset
count=0
for key,value in data_dict.items():
    if value["poi"]==True:
        count=count+1
print count
## dataset contain 18 poi person
count=0
# print the NaN of each feature in data_dict
### Task 2: Remove outliers
data_dict.pop("TOTAL")
# "TRAVEL AGENCY IN THE PARK" is nob a person ,romove it
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
# "LOCKHART EUGENE E" is the person all the values is NaN, remove it 
data_dict.pop("LOCKHART EUGENE E")
### Task 3: Create new feature(s)
# a new feature define total_stock total_messages=exercised_stock_options+restricted_stock
total_messages=[]
salary=[]
total_payments=[]
bonus=[]
total_stock_value=[]
expenses=[]
exercised_stock_options=[]
other=[]
long_term_incentive=[]
restricted_stock=[]
to_messages=[]
from_poi_to_this_person=[]
from_messages=[]
from_this_person_to_poi=[]
shared_receipt_with_poi=[]
def removeNaN(item):
    if item=="NaN":
        item=0
    return item
for key,value in data_dict.items():
   message=removeNaN(value['to_messages'])+removeNaN(value['from_messages'])
   total_messages.append(message)
   salary.append(removeNaN(value["salary"]))
   total_payments.append(removeNaN(value["total_payments"]))
   bonus.append(removeNaN(value["bonus"]))
   total_stock_value.append(removeNaN(value["total_stock_value"]))
   expenses.append(removeNaN(value["expenses"]))
   exercised_stock_options.append(removeNaN(value["exercised_stock_options"]))
   other.append(removeNaN(value["other"]))
   long_term_incentive.append(removeNaN(value["long_term_incentive"]))
   restricted_stock.append(removeNaN(value["restricted_stock"]))
   to_messages.append(removeNaN(value["to_messages"]))
   from_poi_to_this_person.append(removeNaN(value["from_poi_to_this_person"]))
   from_messages.append(removeNaN(value["from_messages"]))
   from_this_person_to_poi.append(removeNaN(value["from_this_person_to_poi"]))
   shared_receipt_with_poi.append(removeNaN(value["shared_receipt_with_poi"]))
### Store to my_dataset for easy export below.
def rescale( orginal_list ):
    reshaplist=[]
    max_number=max(orginal_list)
    min_number=min(orginal_list)
    for i in orginal_list:
        data=float(i-min_number)/float(max_number-min_number)
        reshaplist.append(data)
    return reshaplist

my_dataset = data_dict
salary=rescale(salary)
total_payments=rescale(total_payments)
bonus=rescale(bonus)
total_stock_value=rescale(total_stock_value)
expenses=rescale(expenses)
exercised_stock_options=rescale(exercised_stock_options)
other=rescale(other)
long_term_incentive=rescale(long_term_incentive)
restricted_stock=rescale(restricted_stock)
to_messagese=rescale(to_messages)
from_poi_to_this_person=rescale(from_poi_to_this_person)
from_messages=rescale(from_messages)
from_this_person_to_poi=rescale(from_this_person_to_poi)
shared_receipt_with_poi=rescale(shared_receipt_with_poi)
total_messages=rescale(total_messages)
count=0
for key,value in data_dict.items():
    item=value
    item["salary"]=salary[count]
    item["total_payments"]=total_payments[count]
    item["bonus"]=bonus[count]
    item["total_stock_value"]=total_stock_value[count]
    item["expenses"]=expenses[count]
    item["exercised_stock_options"]=exercised_stock_options[count]
    item["other"]=other[count]
    item["long_term_incentive"]=long_term_incentive[count]
    item["restricted_stock"]=restricted_stock[count]
    item["to_messagese"]=to_messagese[count]
    item["from_poi_to_this_person"]=from_poi_to_this_person[count]
    item["from_messages"]=from_messages[count]
    item["from_this_person_to_poi"]=from_this_person_to_poi[count]
    item["shared_receipt_with_poi"]=shared_receipt_with_poi[count]
    item["total_messages"]=total_messages[count]
    my_dataset[key]=item
    count=count+1

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest,f_classif

#selector=SelectKBest(f_classif, k=3)
#selector.fit(features,labels)
#features=selector.transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn import tree
from sklearn.svm import SVC
#clf = GaussianNB()
treeclf=tree.DecisionTreeClassifier()
parameters = {'criterion':('gini','entropy'),'min_samples_split':[2,10],'presort':(1,0)}
clf=GridSearchCV(treeclf,parameters)
clf.fit(features, labels)
print clf.best_params_
clf= tree.DecisionTreeClassifier(min_samples_split=2, presort=1, criterion='entropy')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
kf=KFold(len(features),4)
for train_indices,test_indices in kf:
    features_train=[features[ii] for ii in train_indices]
    features_test=[features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
#features_train, features_test, labels_train, labels_test = \
    #train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(pred,labels_test)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision=precision_score(pred,labels_test)
recall=recall_score(pred,labels_test)
print accuracy
print precision
print recall




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)