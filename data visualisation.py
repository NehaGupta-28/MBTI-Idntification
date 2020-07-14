#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load('en')
import re
from collections import Counter
df = pd.read_csv("sigh/JudPer.csv")
posts = df['posts'].values.tolist()
trait = df['trait'].values.tolist()
df['posts'] = df['posts'].replace(to_replace = r'\|\|\|', value = r';',regex=True)
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url'
df['posts'] = df['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)
pers_type = ['Judging' ,'Perceiving']
pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
p = re.compile("(" + "|".join(pers_types) + ")")
df['posts'] = df['posts'].replace(to_replace = p, value = r'PTypeToken', regex = True)
total_dict = {}
for i in range(len(posts)):
    total_dict[i] = [trait[i],posts[i]]

trait = df.groupby('trait').count()
trait.sort_values("trait", ascending=False, inplace=True)

def bag_of_words(group, type_label):
    posts = [t for t in group.get_group(type_label)['posts']]
    nlp = spacy.load('en_core_web_sm')
    count_tags = Counter()
    for posts_per_user in posts:
        doc = nlp(str(posts_per_user))
        count_tags.update(Counter([token.pos_ for token in doc]))
    return count_tags
def tags_pie_plot(count_tags):
    bag_of_tags = list(count_tags.keys())
    bag_of_tags_values = [count_tags.get(l) for l in bag_of_tags[:5]]
    fig = plt.pie(bag_of_tags_values, labels = bag_of_tags[:5], autopct = '%1.1f%%', startangle = 140)
    return fig
types_grouped = df.groupby('trait')
for t in pers_type:
    count_tags = bag_of_words(types_grouped, t)
    tags_pie_plot(count_tags)
    print(t)
    plt.show()