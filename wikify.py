#!/usr/bin/env python3

import gzip
import os
import fnmatch
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import wikipedia
from nltk.tag.stanford import StanfordNERTagger


def set_path():
    """Returns the path we need to work in."""
    return '/home/s2958058/pta/test/p16/'


def get_sentences():
    """Returns the sentences as a list"""
    path = set_path()
    configfiles = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in fnmatch.filter(files, '*.tok.off.pos')]
    sentences = []
    for fname in sorted(configfiles):
        with open(fname) as infile:
            for line in infile:
                line = line.split()
                if len(line) == 5:
                    sentences.append(line)
    return sentences


def get_text():
    """Gets the text from the raw text files"""
    all_text = []
    path = set_path()
    configfiles = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in fnmatch.filter(files, '*.raw')]
    for fname in sorted(configfiles):
        with open(fname) as infile:
            for line in infile:
                line = line.strip()
                line = line.strip('.')
                if line != "":
                    all_text.append(line)
    return [item for item in all_text]


def get_nouns():
    """Returns only nouns"""
    sentences = get_sentences()
    return [item[3] for item in sentences if item[4] == 'NNP' or item[4] == 'NN' or item[4] == 'NNS']


def one_line():
    return ' '.join(get_text())


def get_named_entities():
    """NNP Chunking attempt to get nouns that belong together together"""
    text = get_text()
    named_entity_list = []
    for sent in text:
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                named_entity_list.append(' '.join(c[0] for c in chunk.leaves()))
    return [name_entity for name_entity in named_entity_list if ' ' in name_entity]


def named_entity_dict():
    """Returns a dictionary with words as keys and the whole entity as values."""
    named_entities = get_named_entities()
    noun_list = get_nouns()
    noun_dict = {}
    for named_entity in named_entities:
        dict_values = named_entity.split()
        noun_dict[tuple(dict_values)] = named_entity
    return noun_dict


def o_tag():
    """Returns a noun with a tag if the tag is unfindable or a location"""
    sttag = StanfordNERTagger('stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner-2014-06-16/stanford-ner-3.4.jar')
    sttags = sttag.tag(get_nouns())
    return [sttag for sttag in sttags if sttag[1] == 'O' or sttag[1] == "LOCATION"]


def ner_tag():
    """Returns a noun with a tag if the tag is person or organization"""
    sttag = StanfordNERTagger('stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner-2014-06-16/stanford-ner-3.4.jar')
    sttags = sttag.tag(get_nouns())
    return [sttag for sttag in sttags if sttag[1] == "PERSON" or sttag[1] == "ORGANIZATION"]


def disambiguate():
    """Returns the best synset if a word is ambiguous"""
    lesks = []
    nouns = o_tag()
    text = one_line()
    for noun in nouns:
        if len(wn.synsets(noun[0])) > 1:
            if lesk(text, noun[0], 'n'):
                lesks.append((noun[0], lesk(text, noun[0], 'n')))
        elif lesk(text, noun[0], 'n'):
            lesks.append((noun[0], wn.synsets(noun[0])))
    return lesks


def unlist(given):
    """This function removes the lists as item from lists"""
    new_list = []
    for item in given:
        if isinstance(item[1], list):
            new_list.append((item[0], item[1][0]))
        else:
            new_list.append(item)
    return new_list


def look_entity():
    """Checks if something is an entity"""
    lesks = unlist(disambiguate())
    tag_dict = {}
    entity_dict = {'country': 'COU', 'district': 'COU', 'city': 'CIT', 'town': 'CIT',  'geological_formation': 'NAT', 'animal': 'ANI', 'sport': 'SPO',  'entertainment': 'ENT', 'body_of_water': 'NAT', 'desert': 'NAT'}
    for item in ner_tag():
        noun = item[0]
        tag_dict[noun] = item[1][:3]
    for synset in lesks:
        noun = synset[0]
        for item in synset[1].hypernym_paths():
            for i in item:
                name = i.name()[:-5]
                if name in entity_dict.keys():
                    name = entity_dict[name]
                    tag_dict[noun] = name
    return tag_dict


def combine():
    """Returns a part of an entity as key with the whole entity as value"""
    tag_dict = look_entity()
    combined = {}
    noun_dict = named_entity_dict()
    for item in tag_dict.keys():
        for key, value in noun_dict.items():
            if item in key:
                combined[item] = value
    return combined


def wikify():
    """Returns the sentences with wikipedia links"""
    tag_dict = look_entity()
    link_dict = {}
    combined = combine()
    for item in tag_dict.keys():
        if item in combined.keys():
            try:
                if combined[item] in wikipedia.page(combined[item]).content:
                    link_dict[item] = wikipedia.page(combined[item]).url
            except wikipedia.exceptions.DisambiguationError as disamb:
                try:
                    link_dict[item] = wikipedia.page(disamb.options[0]).url
                except:
                    pass
            except wikipedia.exceptions.PageError:
                pass
        else:
            try:
                link_dict[item] = wikipedia.page(item).url
            except wikipedia.exceptions.DisambiguationError as disamb:
                try:
                    link_dict[item] = wikipedia.page(disamb.options[0]).url
                except:
                    pass
            except wikipedia.exceptions.PageError:
                pass
    return link_dict


def output():
    """Writes the sentences with their tags and links into a file"""
    links = wikify()
    tags = look_entity()
    path = set_path()
    configfiles = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in fnmatch.filter(files, '*.tok.off.pos')]
    for fname in sorted(configfiles):
        outfile = open(fname + '.ent', 'w')
        with open(fname) as infile:
            for item in infile:
                item = item.split()
                for key, value in tags.items():
                    if item[3] == key:
                        item.append(value)
                for key, value in links.items():
                    if item[3] == key:
                        item.append(value)
                outfile.write(' '.join(item) + '\n')


def main():
    print(look_entity())
if __name__ == '__main__':
    main()

