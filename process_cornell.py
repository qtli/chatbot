import ast
import os

from nltk import word_tokenize

from utils import save_data

CORPUS_PATH = 'data/cornell movie-dialogs corpus'
OUTPUT_PATH = 'data/processed/cornell.txt'

ENCODING = 'ISO-8859-2'


def get_cornell_data():
    data = []
    for c in get_conversations():
        num_movie_lines = len(c)
        for i in xrange(num_movie_lines - 1):
            qa_pair = (c[i], c[i + 1])
            data.append(qa_pair)
    return data


def get_conversations():
    conversations_path = os.path.join(CORPUS_PATH, 'movie_conversations.txt')
    conversations = load_raw_data(conversations_path)
    id_to_line = get_id_to_line()
    for c in conversations:
        line_id_list = c[-1]  # represented as a string
        yield [id_to_line[line_id] for line_id in ast.literal_eval(line_id_list)]


def get_id_to_line():
    lines_path = os.path.join(CORPUS_PATH, 'movie_lines.txt')
    lines = load_raw_data(lines_path)
    id_to_line = {}
    for l in lines:
        line_id = l[0]
        line_text = l[-1]
        id_to_line[line_id] = line_text.decode(ENCODING)
    return id_to_line


def load_raw_data(file_path):
    seperator = ' +++$+++ '
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip().split(seperator)


def tokenize(text):
    return [word for word in word_tokenize(text.lower()) if not is_number(word)]


# https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
def is_number(word):
    try:
        float(word)
    except ValueError:
        return False
    return True


if __name__ == '__main__':
    print 'Getting data...'
    data = get_cornell_data()

    print 'Tokenizing...'
    data = [(tokenize(q), tokenize(a)) for q, a in data]

    print 'Saving...'
    save_data(data, ENCODING, OUTPUT_PATH)
