#@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
import re
import pandas as pd
import numpy as np
import math
from collections import Counter
from transformers import pipeline
import operator
import string


def percentage(part, whole):
    return 100 * float(part) / float(whole)

def count_values_in_column(data,feature):
  total=data.loc[:,feature].value_counts(dropna=False)
  percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
  return pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])


def remove_punctuation(text):
    text_no_punc = "".join([c for c in text if c not in string.punctuation])
    return text_no_punc