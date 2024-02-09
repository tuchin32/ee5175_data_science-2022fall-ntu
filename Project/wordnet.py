from nltk.corpus import wordnet as wn
print(wn.synsets('draft'))
print(wn.synsets('draft')[0].definition())
