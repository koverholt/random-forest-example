#!/usr/bin/env python

# This example is from
# http://earlbellinger.wordpress.com/2013/12/14/random-forests/

from random import randint, uniform, gauss
from numpy import asarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

num_chairs = num_tables = 5000

data = [[randint(0, 5),  # Possible colors for chairs
         uniform(2, 5),  # Possible leg lengths for chairs
         gauss(2, 0.25)]  # Possible top surface areas for chairs
        for i in range(num_chairs)] + \
       [[randint(0, 5),  # Possible colors for tables
         uniform(4, 10),  # Possible leg lengths for tables
         gauss(5, 1)]  # Possible top surface areas for tables
        for i in range(num_tables)]

labels = asarray(['chair']*num_chairs + ['table']*num_tables)

rfc = RandomForestClassifier(n_estimators=100)
# rfc.fit(data, labels)

scores = cross_val_score(rfc, data, labels, cv=10)
print('Accuracy: %0.2f +/- %0.2f' % (scores.mean(), scores.std()*2))
