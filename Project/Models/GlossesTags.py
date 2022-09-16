import os
from flair.data import Sentence
from flair.models import SequenceTagger

DATA_PATH = os.path.join("Data", "Phoenix14")

questions_words = ["wo", "woher", "wohin", "wann", "was", "wer", "wie", "warum", "wenn", "etwas"]
# pronouns_words = ["ich", "du", "sei", "er", "sie", "IHR"]
months = ['oktober', 'november', 'juli', 'februar', 'dezember', 'januar', 'mai', 'september', 'april', 'august', 'maerz', 'juni']
# season = ['herbst', 'sommer', 'winter', 'fruehling']
days = ['montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag', 'sonntag']
locations = ['england', 'schottland', 'bremen', 'nordpol', 'brandenburg', 'allgaeu', 'polen', 'eifel', 'island', 'schweden', 'bodensee', 'muenster', 'nordpol', 'rumaenien', 'ungarn', 'weser', 'deutschland', 'schottland', 'amerika']
numbers = ['zeh', 'fuenf', 'fuenfzehn', 'zwoelf', 'sechszehn', 'hundert', 'fuenfzig', 'dreissig', 'siebte', 'erste', 'zweite', 'sechste', 'zehnte', 'dritte', 'sechshundert', 'fuenfhundert', 'dreihundert', 'zwoelfte', 'elfte', 'elf', 'neunzehnte', 'sechzig', 'vierte', 'achte', 'fuenfte', 'neunte', 'zuerst', 'erst']

TAGS_LIST = ['NULL', 'LTR', 'QST', 'MNT', 'DAY', 'LOC', 'NUM']


def get_glosses_tags(gloss_vocab):
    glosses_idx_list = gloss_vocab.get_itos()

    tagger_ner = SequenceTagger.load("flair/ner-german-large")
    tagger_pos = SequenceTagger.load("flair/upos-multi")

    gloss_tag = {}
    for i in range(len(glosses_idx_list)):
        gloss_name = glosses_idx_list[i]
        if gloss_name in gloss_vocab.specials_tokens:
            gloss_tag[i] = 'NULL'
        if len(gloss_name) == 1 or gloss_name == 'SCH' or gloss_name == "NN" or gloss_name == "MM":
            gloss_tag[i] = 'LTR'
            continue
        if gloss_name.lower() in questions_words:
            gloss_tag[i] = 'QST'
            continue
        if gloss_name.upper() in months:
            gloss_tag[i] = 'MNT'
            continue
        if gloss_name.lower() in days:
            gloss_tag[i] = 'DAY'
            continue
        if gloss_name.upper() in locations:
            gloss_tag[i] = 'LOC'
            continue
        if gloss_name.upper() in numbers:
            gloss_tag[i] = 'NUM'
            continue
        tmp = gloss_name.split("-")
        if len(tmp) == 2 and (tmp[0].lower() == "neg" or tmp[0].lower() == "nicht"):
            gloss = Sentence(tmp[1].lower())
        else:
            gloss = Sentence(gloss_name.lower())
        tagger_ner.predict(gloss)
        tagger_pos.predict(gloss)
        if 'ner' in gloss.annotation_layers.keys() and gloss.annotation_layers['ner'][0].value == 'LOC':
            gloss_tag[i] = 'LOC'
        else:

            if gloss.annotation_layers['upos'][0].score > 0.7 and gloss.annotation_layers['upos'][0].value in TAGS_LIST:
                gloss_tag[i] = gloss.annotation_layers['upos'][0].value
            else:
                gloss_tag[i] = "NULL"

    counter = {}
    for k, v in gloss_tag.items():
        if v not in counter.keys():
            counter[v] = []
        counter[v].append(k)

    return gloss_tag


