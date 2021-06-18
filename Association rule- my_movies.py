# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:41:47 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
movies=pd.read_csv("my_movies.csv")
movies= movies.iloc[:,5:]

#Apriori algorithm
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets=apriori(movies,min_support=0.05,max_len=3,use_colnames=True)
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#plot
plt.bar(x = list(range(1,11)), height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11]);
plt.xlabel('item sets');plt.ylabel('support')

#Rules
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)


############# To eliminate the Redundancy in Rules #########################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)


unique_rules_sets = [list(m) for m in set (tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

#Getting the rules without any redundancy
rules_no_redundancy = rules.iloc[index_rules,:]

#Sorting them with respect to list and getting top 10 rules
rules_no_redundancy.sort_values('lift', ascending=False).head(10)























