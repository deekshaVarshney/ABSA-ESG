import os
import math
import torch
import numpy as np
import pandas as pd
import random
import warnings
import datetime
from scipy import stats
warnings.filterwarnings('ignore')
import json
import re
from nltk.stem import WordNetLemmatizer
from functools import reduce
import stanza
import csv
import nltk
import logging
import re
import nltk
from tqdm import tqdm as tqdm_bar

from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and numbers using regular expressions
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join the words back into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

print("GPU Available: {}".format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
try:
  print("Number of GPU Available: {}".format(n_gpu))
  print("GPU: {}".format(torch.cuda.get_device_name(0)))
except:
  pass

# stanza.download_pipeline('en') # download English model
nlp = stanza.Pipeline('en', use_gpu = True) # initialize English neural pipeline
doc = nlp("what is cio views for client 00001") # run annotation over a sentence, annotations include tokenization, POS tag, NER, dependency parsing, etc.
print(doc)

wordnet_lemmatizer = WordNetLemmatizer()

class concept_parser_v2:
  """
  Class containing all word pair combination functions.
  """

  def __init__(self, words, postags):
    """
    Initializes the class with words and postags.

    Args:
    - words: list of words
    - postags: list of part-of-speech tags
    """
    self.words = words
    self.postags = postags
    self.ignore = 0
    pass

  def nsubject(self,line):
    """
    Returns a list of nominal subjects in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of nominal subjects
    """
    final_concept = []
    final_nsub = []
    pos = []

    # line is like the extracted relationship between two particular words, which is the input
    # it will look like nsubj, formalise, we (we is the nominal subject of formalise)
    # this is to append both the words and their post tags into the final_nsub and pos list
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_nsub:
          final_nsub.append(self.words[j])
          pos.append(self.postags[j])
    if len(final_nsub) > 1:

      # if there is no Determiner within the position (i.e. words like 'the', 'a', 'an', 'this'))
      if "DT" not in str(pos):

        # if noun or adjective is in the POS..
        if "NN" in str(pos) or "JJ" in str(pos):

          if "JJ" in str(pos):
            # Append the adjective first, then the noun
            final_concept.append(final_nsub[0] + "_" + final_nsub[1])

          # IF A NOUN IS IN THE POS
          if "NN" in str(pos):
            # AND IF A PERSONAL PRONOUN IS IN THE POS
            if "PRP" in str(pos):
              # APPEND ONLY THE NOUN
              final_concept.append(final_nsub[0])

            else:
              # append the noun then the other word
              final_concept.append(final_nsub[1] + "_" + final_nsub[0])

        if "NN" not in str(pos):
          if "PRP" in str(pos):
              pass

          else:
              pass
        
      # if Determiner is inside
      if "DT" in str(pos):
        # append only the noun
        final_concept.append(final_nsub[0])

    if not final_concept:
      pass
    else:
      return final_concept

  def prt(self,line):
    """
    Returns a list of particles in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of particles
    """
    final_concept = []
    final_prt = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_prt:
          final_prt.append(self.words[j])
    
    if len(final_prt) > 1:
      final_concept.append(final_prt[0] + "_" + final_prt[1])

    if not final_concept:
      pass
    else:
      return final_concept

  def casef(self,line):
    """
    Returns a list of case frames in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of case frames
    """
    final_concept = []
    final_case = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_case:
          final_case.append(self.words[j])

    if len(final_case) > 1:
      if final_case[1] in ['from', 'to']:
        final_concept.append(final_case[1] + "_" + final_case[0])

    if not final_concept:
      pass
    else:
      return final_concept

  def acl(self,line):
    """
    Returns a list of acl (adjectival clause) in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of acl
    """
    final_concept = []
    final_acl = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_acl:
          final_acl.append(self.words[j])

    if len(final_acl) > 1:
      final_concept.append(final_acl[1] + "_" + final_acl[0])

    if not final_concept:
      pass
    else:
      return final_concept

  def det(self,line):
    """
    Returns a list of determiners in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of determiners
    """
    final_concept = []
    final_det = []
    pos = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_det:
          final_det.append(self.words[j])
          pos.append(self.postags[j])

    if len(final_det) > 1:

      if "DT" not in str(pos):
        final_concept.append(final_det[1] + "_" + final_det[0])
      if "DT" in str(pos):
        final_concept.append(final_det[0])

    if not final_concept:
      pass
    else:
      return final_concept

  def dep(self,line):
    """
    Returns a list of dependent in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of dependent
    """
    final_concept = []
    final_dep = []
    pos = []

    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_dep:
          final_dep.append(self.words[j])
          pos.append(self.postags[j])

    if len(final_dep) > 1:

      if "DT" not in str(pos) and "JJ" in str(pos):
        final_concept.append(final_dep[1] + "_" + final_dep[0])

      if "DT" not in str(pos) and "JJ" not in str(pos):
        if "NN" in str(pos) and "VB" not in str(pos):
          final_concept.append(final_dep[0])
        else:
          final_concept.append(final_dep[0] + "_" + final_dep[1])

      if "DT" in str(pos):
        final_concept.append(final_dep[0])

    if not final_concept:
      pass
    else:
      return final_concept

  def obj(self,line):
    """
    Returns a list of direct objects in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of direct objects
    """
    final_concept = []
    final_obj = []

    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_obj:
          final_obj.append(self.words[j])
    if len(final_obj) > 1:
      final_concept.append(final_obj[0] + "_" + final_obj[1])

    if not final_concept:
      pass
    else:
      return final_concept

  def dobj(self,line):
    """
    Returns a list of direct objects in the given line.

    Args:
    - line: list of words

    Returns:
    - final_concept: list of direct objects
    """
    final_concept = []
    final_dobj = []

    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_dobj:
          final_dobj.append(self.words[j])
    if len(final_dobj) > 1:
      final_concept.append(final_dobj[0] + "_" + final_dobj[1])

    if not final_concept:
      pass
    else:
      return final_concept
  # acomp : adjectival complement : Adjectival complement of a verb is an adjectival phrase which functions as the complement
  def acomp(self,line):
    final_concept = []
    final_acomp = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_acomp:
          final_acomp.append(self.words[j])
    if len(final_acomp) > 1:
      final_concept.append(final_acomp[0] + "_" + final_acomp[1])

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept

  # advmod : adverbial modifier : Adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase (ADVP) that serves to modify the meaning of the word
  def advmod(self,line):
    final_concept = []
    final_advmod = []
    pos = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_advmod:
          final_advmod.append(self.words[j])
          pos.append(self.postags[j])

    if len(final_advmod)>1:
    #print pos
      if "VB" in str(pos) and "JJ" in str(pos):
        final_concept.append(final_advmod[0] + "_" + final_advmod[1])
      if "VB" in str(pos) and "JJ" not in str(pos) and "IN" in str(pos):
        final_concept.append(final_advmod[0] + "_" + final_advmod[1])
      if "VB" in str(pos) and "JJ" not in str(pos) and "IN" not in str(pos):
        final_concept.append(final_advmod[1] + "_" + final_advmod[0])
      if "VB" not in str(pos):
        final_concept.append(final_advmod[1] + "_" + final_advmod[0])

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept

  # amod : adjectival modifier : Adjectival modifier of an NP is any adjectival phrase that serves to modify the meaning of the NP
  def amod(self,line):
    final_concept = []
    final_amod = []
    pos = []
    #print line
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_amod:
          final_amod.append(self.words[j])
          pos.append(self.postags[j])
    if len(final_amod) > 1:
      if "VB" in str(pos):
        final_concept.append(final_amod[0] + "_" + final_amod[1])
      else:
        final_concept.append(final_amod[1] + "_" + final_amod[0])

    #print pos

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept
    
  def nmod(self,line):
    final_concept = []
    final_nmod = []
    pos = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_nmod:
          final_nmod.append(self.words[j])
          pos.append(self.postags[j])
    if len(final_nmod) > 1:
      if "VB" in str(pos):
        # if verb is present, put it in front of the entity?
        final_concept.append(final_nmod[0] + "_" + final_nmod[1])
      else:
        final_concept.append(final_nmod[1] + "_" + final_nmod[0])

    #print pos

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept
    
  def apposmod(self,line):
    final_concept = []
    final_appos = []
    pos = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_appos:
          final_appos.append(self.words[j])
          pos.append(self.postags[j])
    if len(final_appos) > 1:
      if "VB" in str(pos):
        # if verb is present, put it in front of the entity?
        final_concept.append(final_appos[0] + "_" + final_appos[1])
      else:
        final_concept.append(final_appos[1] + "_" + final_appos[0])

    #print pos

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept
  

  # aux : auxiliary : Auxiliary of a clause is a non-main verb of the clause
  def aux(self,line):
    final_concept = []
    final_aux = []
    pos = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_aux:
          final_aux.append(self.words[j])
          pos.append(self.postags[j])
    if len(final_aux) > 1:
      if "TO" in pos:
        final_concept.append(final_aux[0])

      else:
        if "VB" in str(pos):
          #print "VB in pos"
          pass
        if "VB" not in str(pos):
          #print "VB not in pos"
          final_concept.append(final_aux[1] + "_" + final_aux[0])

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept

  # nn : noun compound modifier : Noun compound modifier of an NP is any noun that serves to modify the head noun
  def nn(self,line):
    final_concept = []
    final_nn = []
    posit_sum = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_nn:
          final_nn.append(self.words[j])
          posit_sum.append(j)
    if len(final_nn) > 1:
      if (posit_sum[0] - posit_sum[1]) == 1:
        final_concept.append(final_nn[1] + "_" + final_nn[0])
      else:
        final_concept.append(final_nn[0] + "_" + final_nn[1])

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept
    
  def compound(self,line):
    final_concept = []
    final_compound = []
    posit_sum = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_compound:
          final_compound.append(self.words[j])
          posit_sum.append(j)
    if len(final_compound) > 1:
      if (posit_sum[0] - posit_sum[1]) == 1:
        final_concept.append(final_compound[1] + "_" + final_compound[0])
      else:
        final_concept.append(final_compound[0] + "_" + final_compound[1])

    if not final_concept:
      pass
    else:
      return final_concept

  def neg(self,line):
    final_concept = []
    final_neg = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_neg:
          final_neg.append(self.words[j])

    if len(final_neg) > 1:

      if final_neg[0] != final_neg[1]:
        final_concept.append(final_neg[1] + "_" + final_neg[0])


    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept

  # prep : prepositional modifier : Prepositional modifier of a verb, adjective, or noun is any prepositional phrase that serves to modify the meaning of the verb, adjective, noun, or even another prepositon
  def prep(self,line):
    final_concept = []
    final_prep = []
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_prep:
          final_prep.append(self.words[j])
    if len(final_prep) > 1:
      final_concept.append(final_prep[0] + "_" + final_prep[1])
    #print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
    #print(final_concept)
    #print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
    if not final_concept:
      pass
    else:
      return final_concept

  def prep_(self,line):
    final_concept = []
    final_prep = []
    final_p = []
    flag = 0
    prep = line[0].split("_")[1]
    for i in line:
      for j in range(0,len(self.words)):
        if i == self.words[j] and self.words[j] not in final_prep:
          final_prep.append(self.words[j])

    if len(final_prep) > 1:
      final_concept.append(final_prep[0] + "_" + prep + "_" + final_prep[1])

      for i in range(0,len(final_prep)):
        for j in range(0,len(self.words)):
          if final_prep[i] == self.words[j] and "NN" != self.words[j]:
            final_p.append(final_prep[0])
            flag = 1
            break
          else:
            pass

      if flag == 1:
        final_prep = final_p
        final_concept.append(final_prep[0] + "_" + prep)
      if flag == 0:
        final_concept.append(final_prep[0] + "_" + prep + "_" + final_prep[1])

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept

    '''if not final_concept:
      pass
    else:
      #print final_concept
      return final_concept'''

  # prepc : prepositional clausal modifier : prepositional clausal modifier of a verb, adjective, or noun is a clause introduced by a preposition which serves to modify the meaning of the verb, adjective, or noun
  def prepc_(self,line):
    final_concept = []
    final_prepc = []
    final_p = []
    flag = 0
    prep = line[0].split("_")[1]
    for i in line:
      for j in range(0,len(self.words)):
        if i in self.words[j] and self.words[j] not in final_prepc:
          final_prepc.append(self.words[j])

    if len(final_prepc) > 1:
      for i in range(0,len(final_prepc)):
        for j in range(0,len(self.words)):
          if final_prepc[i] in self.words[j] and "NN" not in self.words[j]:
            final_p.append(final_prepc[0])
            flag = 1
            break
          else:
            pass

      if flag == 1:
        final_prepc = final_p
        final_concept.append(final_prepc[0] + "_" + prep)
      if flag == 0:
        final_concept.append(final_prepc[0] + "_" + prep + "_" + final_prepc[1])

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept
    
  def obl_(self,line):
    "Line is a tuple of dependencies"

    final_concept = []
    final_obl = []
    final_o = []
    flag = 0
    obl = line[2]
    for i in line:
      for j in range(0,len(self.words)):
        if i in self.words[j] and self.words[j] not in final_obl:
          final_obl.append(self.words[j])

    if len(final_obl) > 1:
      for i in range(0,len(final_obl)):
        for j in range(0,len(self.words)):
          # if any of the words in the final_obl is in the self.words and the postag is not NN
          if final_obl[i] in self.words[j] and "NN" not in self.postags[j]:
            final_o.append(final_obl[i])
            flag = 1
            break
          else:
            pass

      if flag == 1:
        final_obl = final_o
        final_concept.append(final_obl[0] + "_" + obl)
      if flag == 0:
        final_concept.append(final_obl[0] + "_" + obl + "_" + final_obl[1])

    if not final_concept:
      pass
    else:
      #print(final_concept)
      return final_concept


  # This rule has been created for "TO" type postags for relation between objects
  def manual(self,words, postags):
    manual_concept = []
    for i in range(0,len(words)-1):
      pos = postags[i-1] + postags[i] + postags[i+1]
      word = words[i-1] + "_" + words[i] + "_" + words[i+1]
      if "JJTOVB" in pos or "JJTOVBD" in pos or "JJTOVBZ" in pos or "JJSTOVB" in pos or "JJSTOVBD" in pos or "JJSTOVBZ" in pos or "JJRTOVB" in pos or "JJRTOVBD" in pos or "JJRTOVBZ" in pos and words[i-1] is not words[i+1]:
        manual_concept.append(word)
      else:
        pass

    if not manual_concept:
      pass
    else:
      #print(final_concept)
      return manual_concept


  def conjugation_finder(self, words, postags):
    count_and = []
    count_or = []
    for i in range(0,len(words)):
      if "and" in words and "CC" in postags[i]:
        count_and = [n for (n, e) in enumerate(words) if e == 'and']

    for i in range(0,len(words)):
      if "or" in words and "CC" in postags[i]:
        count_or = [n for (n, e) in enumerate(words) if e == 'or']

    for i in count_and:
      if "CC" in postags[i]:
        pass
      else:
        count_and.remove(int(i))

    for i in count_or:
      if "CC" in postags[i]:
        pass
      else:
        count_or.remove(int(i))

    count_conj = count_and + count_or
    count_conj.sort()
    return count_conj

  # This rule has been created for "AND" types for relation between structures of sentence
  def conjugator(self, words, postags, position):
    conju = ""
    final_concept = []
    flag = 0
    flag1 = 0
#   for i in range(0,len(words)):
#     if "and" in words[i] and "CC" in postags[i]:
#       and_pos = i
#       conju = words[i]
#     if "or" in words[i] and "CC" in postags[i]:
#       and_pos = i
#       conju = words[i]
    and_pos = position
    conju = words[position]
    word1 = and_pos - 1


    for i in range(and_pos, len(words)):
      if "DT" not in postags[i]:
        word2 = and_pos + (i - and_pos)

    verb = ""
    noun = ""

    for i in range(and_pos - 3,and_pos):
      if "VB" in postags[i]:
        verb = words[i]
        flag = 1
      if "NN" in postags[i]:
        noun = words[i]
        flag1 = 1
    #print "************************************************"
    #print and_pos, len(words)
    #print "words[word1]" + words[word1]
    #print "words[word2]" + words[word2]
    #print "verb" + verb
    #print "noun" + noun
    if flag1 == 1:
      for i in range(and_pos,len(words)):
        if noun != "":
          #print "HERE"
          if words[word1] != noun and words[word2] != noun:
            #print "Not noun"
            final_concept.append(words[word1] + "_" + noun)
            final_concept.append(words[word2] + "_" + noun)

          if words[word1] == noun:
            #print "Noun1"
            final_concept.append(words[word2] + "_" + noun)

          if words[word2] == noun:
            #print "Noun2"
            final_concept.append(words[word1] + "_" + noun)
          break
        if "between" in str(words[i]) or "over" in str(words[i]) or "with" in str(words[i]) or "on" in str(words[i]) or "to" in str(words[i]) or "of" in str(words[i]) or "into" in str(words[i]) or "in" in str(words[i]) or "at" in str(words[i]):
          word3 = i + 1
          if words[word1] is not words[word3]:
            final_concept.append(words[word1] + "_" + words[word3])
          if words[word2] is not words[word3]:
            final_concept.append(words[word2] + "_" + words[word3])
          #flag = 1
          break
    if flag == 1:
      #print "NOT HERE"
      final_concept.append(verb + "_" + words[word1])
      final_concept.append(verb + "_" + words[word2])
      final_concept.append(words[word1] + "_" + conju + "_" + words[word2])
    #print "************************************************"
    #print final_concept
    #print "************************************************"
    if not final_concept:
      pass
    else:
      self.ignore = 1
      #print(final_concept)
      return []
    
def append_adjective(i, noun, final_concepts, dependencies, postags, words):
    # Recursively call append_adjective to check for modifiers of the modifying word
    for relation, head, modifying_word in dependencies:
        if head == noun and (relation in ["compound", "amod","nn", "appos", "nummod","flat"] or "nmod" in relation):
            mod_word_idx = words.index(modifying_word)
            if postags[mod_word_idx] in ["JJ","JJR","JJS","NN","NNP","NNPS","POS"]:
              if modifying_word.lower() in (word.lower() for word in final_concepts[i]):
                continue
              final_concepts[i].append(modifying_word)
              # Recursively call append_adjective to check for modifiers of the modifying word
              append_adjective(i, modifying_word, final_concepts, dependencies, postags, words)
    return final_concepts

'''***************************************************************************** MAIN FUNCTION*************************************************************************************'''

"""
nsubj: This stands for "nominal subject." It refers to the noun phrase that is the syntactic subject of a verb. 
For example, in the sentence "The cat sat on the mat," "The cat" is the nominal subject of the verb "sat."

det: This stands for "determiner." Determiners are words that introduce nouns and specify their quantity, identity, or definiteness. 
Examples of determiners include "a," "an," "the," "this," "that," "these," "those," "my," "your," etc.

dep: This stands for "dependent." In dependency grammar, a dependent is a word that is syntactically governed by another word. 
It's a generic relation used when a more specific relation isn't determined.

dobj: This stands for "direct object." It's the noun phrase that receives the action of a transitive verb. 
For instance, in "She reads a book," "a book" is the direct object of the verb "reads."

advmod: This stands for "adverbial modifier." It's an adverb or adverbial phrase that modifies a verb, adjective, or other adverb. 
For example, in "He runs quickly," "quickly" is an adverbial modifier of the verb "runs."

amod: This stands for "adjectival modifier." It's an adjective that modifies a noun. 
For instance, in "red ball," "red" is an adjectival modifier of "ball."

aux: This stands for "auxiliary." Auxiliaries are helping verbs that accompany the main verb 
to express tense, mood, voice, etc. Examples include "is," "was," "have," "will," etc.

nn: This stands for "noun compound modifier." It refers to the situation where one noun 
modifies another noun, forming a compound noun. For example, in "chicken soup," "chicken" is a noun compound modifier of "soup."

prep: This stands for "preposition." Prepositions are words that express relations in space, 
time, or other dimensions, such as "in," "on," "under," "after," "before," etc.

prepc: This stands for "prepositional clausal modifier." It refers to a clause introduced 
by a preposition that modifies a verb, adjective, or noun.

manual: In the context of NLP, "manual" typically refers to tasks or annotations done by humans 
rather than by automated processes. For instance, manual annotation involves humans marking up text with specific labels.

conjugator: This isn't a standard term in dependency parsing, but in the context of linguistics and NLP, 
a conjugator is a tool or resource that provides conjugations for verbs. Conjugation refers to the 
variation of the form of a verb based on tense, mood, person, number, etc.

compound: This relation is used for compound nouns or names. A compound modifier is a modifier that includes more than one word and functions as a single modifier. It's used to link together two words that jointly modify the meaning of another word. For example, in "ice cream," "ice" is a compound modifier for "cream."
    Example Sentence: "I have a credit card payment."
    Parsing: "credit" (compound) <- "card"

nmod:poss: This stands for "nominal modifier: possessive." It is used to indicate possessive relationships in noun phrases. The possessive modifier is typically a possessive pronoun or a noun's possessive case.
    Example Sentence: "This is John's book."
    Parsing: "John's" (nmod:poss) <- "book"

obl: In the context of dependency parsing in natural language processing (NLP), obl stands for "oblique nominal." It is a grammatical relation used to mark non-core 
(or oblique) arguments of a verb that are not subjects or objects. The oblique nominal typically corresponds to nouns or noun phrases that are introduced by 
prepositions, or sometimes postpositions, in languages that use them.
"""

def main(senten):

  senten = senten.strip()
  
  doc = nlp(senten) 
  
  dependencies = []
  words = []
  postags= []


  # Obtaining the dependencies
  for sentence in doc.sentences:
      for word in sentence.words:
          words.append(word.lemma) #lemmatising the words in the sentence and appending them to a list
          postags.append(word.xpos) # tagging the part of speech in the sentence and appending them

          # word.deprel == dependency relation (see above str: nn, prep, prepc, manual, conjugator etc...)
          # word.head == head is the index of the other word that has a direct syntactical relationship with the current word
          dependencies.append(tuple([word.deprel, sentence.words[word.head-1].lemma if word.head > 0 else "ROOT", word.lemma])) 

  # print(dependencies)                                      #Uncomment for the dependencies
  # print(words)                                                                                #Uncomment for the words
  # print(postags)                                         #Uncomment for the postags

  final_concepts = []
  dependency = [] # to store the dependencies after removing the dependencies with PRP and NN

  to_remove = []
  new_flag = 0

  to_delete = []

  for i in range(0,len(dependencies)):

    check_words = []
    check_pos = []

    # OBTAIN WORDS FROM DEPENDENCIES (I.E. NSUBJ FORMALISE, WE --> EXTRACT FORMALISE WE INTO check_words)
    for j in range(0,len(words)):    # Direct removal of PRP (Pronoun) without NN (noun) word pairs (except nsubj)
      # remove_words correspond to the list of words in the sentence, whereas remove corresponds to the list of postags
      if words[j] in dependencies[i][1]:
        check_words.append(dependencies[i][1])
        check_pos.append(postags[j])
        continue
      if words[j] in dependencies[i][2]:
        check_words.append(dependencies[i][2])
        check_pos.append(postags[j])
        continue

    # REMOVE IF FOR TWO WORDS, THE DEPENDENCY IS JUST PRP (preposition) WITHOUT THE NOUNS
    if "PRP" in check_pos and "NN" not in check_pos:
      to_remove.append(dependencies[i])

    if "PRP" in check_pos and "NN" in check_pos:
      to_delete.append(i)     #flag if PRP and NN are present in the same dependency (i.e. PRONOUN + NOUN), which represents the need for removal


  
  # Reverse the list to delete from the end
  to_delete = to_delete[::-1]
  for i in to_delete:
    del dependencies[i]
    
    ## NOTE: For now, do not append nouns to the final_concepts 
    # for k in range(0,len(words)):
    #   # the final concepts will apparently contain nouns for now
    #   if "NN" in postags[k]:
    #     final_concepts.append([words[k]])

  # FILTERING to obtain a to_remove list that contains only the dependencies that are to be removed
  # to_remove_lst = []
  # for i in range(0,len(to_remove)):
  #   to_remove_lst.append(to_remove[i][2] + "_" + to_remove[i][1])


  # CHECK THE DEPENDENCIES --> IF THEY ARE IN THE TO_REMOVE LIST, REMOVE THEM
  if not to_remove:
    pass
  else:
    dependency = []
    for i in dependencies:
      for j in to_remove:
        if i is not j:
          dependency.append(i)
          
        else:
          pass

    dependencies = dependency

  if len(to_remove) > 1:
    # KEEP DEPENDENCIES THAT HAVE GREATER THAN ONE LENGTH (i.e. more than one word)
    dependencies = [i for i in dependencies if dependencies.count(i) > 1]

  # removal of duplicates in the dependencies list
  dependencies = reduce(lambda x, y: x + y if y[0] not in x else x, map(lambda x: [x],dependencies))     


  ## EXTRACTION OF CONCEPTS FROM DEPENDENCIES, THE ABOVE IS CLEANING PART

  concept = concept_parser_v2(words, postags)               # Class Initialization, will contain all the words and postags within the sentence
  for i in range(0,len(dependencies)):
    if "nsubj" in dependencies[i] and "nsubjpass" not in dependencies[i]:     # <function_name> Call
      # print("nsubj", dependencies[i])
      final_concepts.append(concept.nsubject(dependencies[i]))      # nsubj Call

    if "case" in dependencies[i]:      # <function_name> Call
      # print("case", dependencies[i])
      final_concepts.append(concept.casef(dependencies[i]))

    if "acl" in dependencies[i]:      # <function_name> Call
      # print("acl", dependencies[i])
      final_concepts.append(concept.acl(dependencies[i]))

    if "prt" in dependencies[i]:      # <function_name> Call
      # print("prt", dependencies[i])
      final_concepts.append(concept.prt(dependencies[i]))

    #if "det" in dependencies[i]:
    #  print("det", dependencies[i])
    #  final_concepts.append(concept.det(dependencies[i]))       # det Call

    if "dep" in dependencies[i]:
      # print("dep", dependencies[i])
      final_concepts.append(concept.dep(dependencies[i]))       # dep Call

    if "obj" in dependencies[i]:
      # print("obj", dependencies[i])
      final_concepts.append(concept.obj(dependencies[i]))      # dobj Call

    if "dobj" in dependencies[i]:
      # print("dobj", dependencies[i])
      final_concepts.append(concept.dobj(dependencies[i]))      # dobj Call

    if "acomp" in dependencies[i]:
      # print("acomp", dependencies[i])
      final_concepts.append(concept.acomp(dependencies[i]))     # acomp Call

    if "advmod" in dependencies[i]:
      # print("advmod", dependencies[i])
      final_concepts.append(concept.advmod(dependencies[i]))      # advmod Call

    if "amod" in dependencies[i]:
      # print("amod", dependencies[i])
      final_concepts.append(concept.amod(dependencies[i]))      # amod Call

    if "aux" in dependencies[i]:
      # print("aux", dependencies[i])
      final_concepts.append(concept.aux(dependencies[i]))       # aux Call

    if "nn" in dependencies[i]:
      #print("nn", dependencies[i])
      final_concepts.append(concept.nn(dependencies[i]))        # nn Call

    if "neg" in dependencies[i]:
      # print("neg", dependencies[i])
      final_concepts.append(concept.neg(dependencies[i]))       # neg Call

    if "prep" in dependencies[i]:
      # print("prep", dependencies[i])
      final_concepts.append(concept.prep(dependencies[i]))      # prep Call

    if "prep_" in dependencies[i][0]:
      # print("prep_", dependencies[i])
      final_concepts.append(concept.prep_(dependencies[i]))     # prep_<> Call

    if "prepc_" in dependencies[i][0]:
      # print("prepc_", dependencies[i])
      final_concepts.append(concept.prepc_(dependencies[i]))      # prepc_<> Call

    if "obl" in dependencies[i][0]:
      # print("obl", dependencies[i])
      final_concepts.append(concept.obl_(dependencies[i]))      # obl_<> Call
    
    if "compound" in dependencies[i][0]:
      # print("compound", dependencies[i])
      final_concepts.append(concept.compound(dependencies[i]))      # compound Call
    
    if "nmod" in dependencies[i][0]:
      # print("nmod", dependencies[i])
      final_concepts.append(concept.nmod(dependencies[i]))      # nmod Call

    # TODO: in the future check if you want to include apposmod
    # if "appos" in dependencies[i][0]:
    #   # print("appos", dependencies[i])
    #   final_concepts.append(concept.apposmod(dependencies[i]))      # apposmod Call

  ### FURTHER PROCESSING OF CONCEPTS: 

  and_list = []
  position = []
  and_occ = [x for x in words][0]
  #print and_occ
  if and_occ=='and' or and_occ=='or':
    #print 'here'
    pass
  else:
    #print 'not here'
    for i in range(0,len(words)):
      if "and" in words[i] and "CC" in postags[i] or "or" in words[i] and "CC" in postags[i]:
        position = concept.conjugation_finder(words,postags)
        break

  if not position:
    pass
  else:
    for i in position:
      #print(i, words, postags)
      and_list = concept.conjugator(words,postags,int(i))       # conjugator Call, If conjunction is present, rearrange it in the form of <word>_<conjunction>_<Noun>
  #print(and_list)
  if concept.ignore == 0:
    final_concepts.append(concept.manual(words,postags))        # manual Call; to parse the sentence for "TO" type postags for relation between objects
  else:
    pass
  #print(final_concepts)
  #print and_list
  if not and_list:
    pass
  else:
    and_list = [x for x in and_list]             # covert utf-8 to ascii
  #print(final_concepts)
  s = []

  for i in range(0,len(final_concepts)):
    if final_concepts[i] is not None:
      s.append(final_concepts[i])

  final_concepts = s
  #print(final_concepts)

  final = []
  for i in range(0,len(final_concepts)):
      final.append(final_concepts[i][0])

  final = [x for x in final]
  #print('final:', final)
  if not and_list:
    pass
  else:
    final_concepts = final + list(set(and_list) - set(final))
  #print(list(set(and_list) - set(final)))
  if not final_concepts:
    pass
  else:
    final_concepts = reduce(lambda x, y: x + y if y[0] not in x else x, map(lambda x: [x],final_concepts))      # Remove Duplicates
  for i in range(0,len(final_concepts)):
    if len(final_concepts[i]) > 1 or final_concepts[i] is not None:
      pass
      #print final_concepts[i]
    else:
      del final_concepts[i];                          #Removing unwanted concepts of single letters
  
  # put all strings in final_concept into a list
  final_concepts = [x for x in final_concepts if x is not None]
  
  # run the loop on the final concepts to capture n-grams nouns, noun_phrases, adjectives + nouns
  for i in range(0,len(final_concepts)):
      if "_" not in final_concepts[i][0]:
        continue
      # obtain the seperate words
      first_word = final_concepts[i][0].split("_")[0]
      second_word = final_concepts[i][0].split("_")[1]

      # check verb and noun combination
      first_word_idx = words.index(first_word)
      second_word_idx = words.index(second_word)

      first_word_pos = postags[first_word_idx]
      second_word_pos = postags[second_word_idx]

      # TODO: INCLUDE ADVERBS AS WELL, NOT JUST THE VERBS
      if first_word_pos in ["NN","NNS","NNP","NNPS"] and second_word_pos in ["VB","VBD","VBG","VBN","VBP","VBZ"]:
        noun = first_word
        verb = second_word
      
      elif second_word_pos in ["NN","NNS","NNP","NNPS"] and first_word_pos in ["VB","VBD","VBG","VBN","VBP","VBZ"]:
        noun = second_word
        verb = first_word

      else:
        continue

      # print("appending_concept")
      final_concepts  = append_adjective(i,noun, final_concepts, dependencies, postags,words)

  # print(final_concepts)
  return final_concepts                             # Final Concepts of current sentence



if __name__ == '__main__':


  # Function to extract concepts from text
  def extract_concepts_from_tweets(csv_files):
      for file in csv_files:
          # Load the CSV file
          df = pd.read_csv(file)
          print(df.columns)
          
          # Clean the tweets
          df['cleaned_text'] = df['samples'].apply(clean_text)

          # Tokenize the cleaned tweets into sentences
          nltk.download('punkt')
          sentences = df['cleaned_text'].apply(nltk.sent_tokenize)

          # Process each cleaned tweet and extract concepts
          sentence_concepts_list = []
          for item in tqdm_bar(df['cleaned_text']):
              sentences = nltk.sent_tokenize(item)
              concepts = []
              for sentence in sentences:
                  sentence_concepts = main(sentence)
                  concepts.append(sentence_concepts)
              sentence_concepts_list.append(concepts)

          # Add the output sentence concepts to the DataFrame
          df['sentence_concepts'] = sentence_concepts_list
          df = df.sample(frac=1, random_state=42)  # Shuffle with a fixed random state for reproducibility
          out_file = str(file.split('.')[0])
          print(out_file)
          # Save the DataFrame to a new CSV file with added concepts
          df.to_csv(f'{out_file}_with_concepts.csv', index=False)

  # Example usage
  csv_files = ['/Data/deeksha/concept_parser/data/tweets/filtered_tweets.csv', '/Data/deeksha/concept_parser/data/news/filtered_sentences.csv']
  extract_concepts_from_tweets(csv_files)
