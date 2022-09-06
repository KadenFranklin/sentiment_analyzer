import numpy as np
import pandas as pd
import pip
from matplotlib import pyplot as plt
from IPython.display import display, Markdown
from nltk import sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from afinn import Afinn

## BOTH OTHER FILES STORED IN THIS REPOSITORY WERE USED AS REFERENCE FOR THIS FILES CREATION


def show_markdown_table(headers: List[str], data: List) -> str:
    s = f"| {' | '.join(headers)} |\n| {' | '.join([(max(1, len(header) - 1)) * '-' + ':' for header in headers])} |\n"
    for row in data:
        s += f"| {' | '.join([str(item) for item in row])} |\n"
    display(Markdown(s))


def poem_line_analyzer(line):
    afinn = Afinn()
    sid = SentimentIntensityAnalyzer()
    headers = ['Word', 'AFINN', 'VADER']
    rows = [[word,
             afinn.score(word),
             sid.polarity_scores(word)]
            for word in line.split(' ')]
    show_markdown_table(headers, rows)


afinn = Afinn()
sid = SentimentIntensityAnalyzer()

headers = ['Line', 'AFINN', 'VADER']
rows = [[line, afinn.score(line), sid.polarity_scores(line)]
        for line in poem_lines_list]
show_markdown_table(headers, rows)

