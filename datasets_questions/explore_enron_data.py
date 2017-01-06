#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'Number of people in the Enron dataset: {0}'.format(len(enron_data))

print 'Number of features per person: {0}'.format(len(enron_data.values()[0]))

pois = [x for x, y in enron_data.items() if y['poi']]
print 'Number of POIs: {0}'.format(len(pois))

print enron_data["PRENTICE JAMES"]["total_stock_value"]

print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print enron_data["LAY KENNETH L"]["total_payments"], enron_data["SKILLING JEFFREY K"]["total_payments"], enron_data["FASTOW ANDREW S"]["total_payments"] 

with_salaries = [x for x, y in enron_data.items() if y['salary'] != 'NaN']
print 'Number of people with quantified salaries: {0}'.format(len(with_salaries))

with_emails = [x for x, y in enron_data.items() if y['email_address'] != 'NaN']
print 'Number of people with email addresses: {0}'.format(len(with_emails))
