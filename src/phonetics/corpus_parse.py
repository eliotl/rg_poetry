import re, unicodedata
import random
from collections import Counter
import numpy as np
import sys
from phonetics.parse_spacy import spacy_parse

from utils.utils import flatten_list
from utils.file_handling import save_pickle, open_file
from apple_phonemes import get_synth, synth_parse


# Need a better split_into_sentences?


# Splits a text into a nested list of paragraphs and sentences using some regex tricks. 
# from https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
def split_into_sentences(text):

    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|\bmr|\bdr|\bmrs|\bms)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co|\binc|\bjr|\bsr)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Za-z][.][A-Za-z][.](?:[A-Za-z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    websites2 = "(https?://\w)[.]"

    text = " " + text + "  "
    text = text.replace("\n","<graf>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(websites2, "\\1<prd>",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    text = re.sub(r'\s*\.\.+', '<elipses><stop>', text)
    text = re.sub(r'\s*!!+', '<bangpses><stop>', text)
    text = re.sub(r'\s*\?\?+', '<qpses><stop>', text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    # if ")" in text: text = text.replace(".)", ").")

    # text = re.sub(r'(\.+)', '\1<stop>', text)
    
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    # text = re.sub(r'(!+)', '\1<stop>', text)
    text = text.replace("!","!<stop>")

    text = text.replace(".<stop>)", ".)<stop>")
    text = text.replace("?<stop>)", "?)<stop>")
    text = text.replace("!<stop>)", "!)<stop>")


#    print(text[5600:10000])

    text = re.sub(r'(^|<stop>|<graf>)\s*([:\"\'“‘”’;\-—–\[\](){}\/\\\.,`«»]|<prd>|.<stop>)+(\s+|<stop>|<graf>)', r'\1\2', text)
    text = re.sub(r'\s+([:\"\'“‘”’;\-—–\[\](){}\/\\\.,`«»]|<prd>|.<stop>)+(?=(\s+|$|<stop>|<graf>))', r'\1', text)
    text = re.sub(r'\s+(..?(<stop>|<graf>))+(?=(\s+|$))', r'\1', text)
    text = re.sub(r'([a-zA-Z]+)([;:,\"])([a-zA-Z]+)', r'\1 \2\3', text)
    # text = text.replace("\'", "'")

    # For dril purposes it would be a good idea to have something that stopped   gamer;c.  from being one word "gamer=cee"   
    # i.e. to seperate words with certain punctuations in the middle 
    # But the Q is, is it like isn;t or like gamer;c


    text = text.replace("<prd>",".")
    text = text.replace("<elipses>", "...")
    text = text.replace("<bangpses>", "!!!")
    text = text.replace("<qpses>", "???")
    grafs = re.split(r" *<graf> *", text)
    sentences = [x.split("<stop>") for x in grafs if len(x) > 0]
    return sentences

def make_grafMap(grafs, pad = 0):
	grafMap = np.cumsum([pad + len(graf) for graf in grafs])
	# MAKE THIS A LIST NOT AN ARRAY!!!!!
	return grafMap

def sent_to_graf(sent_index, grafMap):
	mod = sent_index + 1
	idx = 0
	while idx < len(grafMap):
		if grafMap[idx] >= mod:
			break
		idx = idx + 1
	i1 = idx
	i2 = max(0,mod - grafMap[idx-1] - 1)
	return [i1, i2]

def graf_to_sent(g_indices, grafMap):
	i1, i2 = g_indices
	if i1 == 0:
		return i2
	idx = grafMap[i1-1]
	return idx + i2


def split_into_paragraphs(text):
    grafs = [[sentence for sentence in graf if len(sentence) > 0] for graf in split_into_sentences(text)]
    sentences = flatten_list(grafs)
    grafMap = make_grafMap(grafs)
    return np.insert(grafMap, 0, 0)

def flender_from_paragraphs(self, wine_grafs):
    def render_scans(i):
        return [(a, b) for a, b in zip(self.wordTokens[i], self.scanTokens[i].split(' '))]
    flendered_graphs = []
    for a, b in zip(wine_grafs, wine_grafs[1:]):
        zip_scans = [f"{bb}_ {aa}" for i in range(a, b) for aa, bb in render_scans(i)]
        flendered_graphs.append(zip_scans)


def flender_from_paragraphs(self, wine_grafs):
    import string
    def render_parses(i):
        return [(a, b) for a, b in zip(self.wordTokens[i], self.parseTokens[i])]
    flendered_graphs = []
    for a, b in zip(wine_grafs, wine_grafs[1:]):
        zip_scans = [f"{b} {a.strip(string.punctuation)}_" for i in range(a, b) for aa, bb in render_parses(i)]
        flendered_graphs.append(zip_scans)


def sync_parse_and_words(wordToken, parseToken):
    # lists of strs
    lenDiff = abs(len(wordToken) - len(parseToken))

    if lenDiff := (len(wordToken) - len(parseToken)) != 0:
    # if lenDiff != 0:
        wordLen = len(wordToken)
        parseLen = len(parseToken)
        bangToken = ' ! '.join(wordToken) + ' !'
        bangParse = synth.phonemesFromText_(bangToken) + ' '
        bangSplit = re.split(r'![!,.:;\']* ', bangParse)
        bangSplit.pop()
        bangSplit = [word.split() for word in bangSplit]
        bangLens = [len(x) for x in bangSplit]
        if len(bangLens) - np.sum(bangLens) == lenDiff:
            newPTs = []
            j = 0
            for k in bangLens:
                if k == 1:
                    newPTs.append(parseToken[j])
                    j = j + 1
                if k > 1:
                    newPT = '='.join(parseToken[j:j+k])
                    newPT = re.sub(r'[^a-zA-Z]*=[^a-zA-Z]*','=', newPT)
                    newPTs.append(newPT)
                    j = j + k
            parseToken = newPTs
            i1, i2 = sent_to_graf(i, grafMap)
            parseTokens0[i1][i2] = newPTs
        else:
            chop = min(len(wordToken), len(parseToken))
            wordToken = wordToken[:chop]
            wordToken = [x + ';;' for x in wordToken]
            parseToken = parseToken[:chop]



def parse_then_sort(text, VERBOSE = True):
    if VERBOSE:
        print('parsing...')
    else:
        if random.random() < 0.001:
            print('1 / 1000')

    parsed_text = spacy_parse(text)

    # grafs = [[sentence for sentence in graf if len(sentence) > 0] for graf in parsed_text.sents]
    sentences = [sentence for sentence in parsed_text.sents]
    # grafMap = make_grafMap(grafs)

    wordTokens0 = [[sentence.split() for sentence in graf] for graf in grafs]
    wordTokens = flatten_list(wordTokens0)

    synth = get_synth()

    parse = []
    ii = 0

    for graf in grafs:
        if ii%1000 == 0:
            if VERBOSE:
                print(str(ii) + ' / ' + str(len(grafs)))
        parse.append([re.sub(r'\[\[.*\]\]', '',synth.phonemesFromText_(sentence)) for sentence in graf])
        ii += 1

    wordTokens0 = [[sentence.split() for sentence in graf] for graf in grafs]
    parseTokens0 = [[sentence.split() for sentence in graf] for graf in parse]

    wordLens0 = [[len(sentence) for sentence in graf] for graf in wordTokens0]
    parseLens0 = [[len(sentence) for sentence in graf] for graf in  parseTokens0]

    wordLarray0 = np.array([np.array(x) for x in wordLens0])
    parseLarray0 = np.array([np.array(x) for x in parseLens0])

    lenDiffs0 = list(np.subtract(wordLarray0, parseLarray0))
    lenDiffs0 = [list(x) for x in lenDiffs0]

    wordTokens = flatten_list(wordTokens0)
    parseTokens = flatten_list(parseTokens0)
    lenDiffs = flatten_list(lenDiffs0)

    count = Counter()
    countP = Counter()
    count.update(lenDiffs)

    if VERBOSE:
       print('sorting...')

    # This is the process of getting them disambiguated

    for i in range(0, len(wordTokens)):

        if i % 1000 == 5:
            if VERBOSE:
                print(i)

        if lenDiffs[i] != 0:
            wordToken = wordTokens[i]
            wordLen = len(wordTokens[i])
            parseToken = parseTokens[i]
            parseLen = len(parseTokens[i])
            bangToken = ' ! '.join(wordToken) + ' !'
            bangParse = synth.phonemesFromText_(bangToken) + ' '
            bangSplit = re.split(r'![!,.:;\']* ', bangParse)
            bangSplit.pop()
            bangSplit = [word.split() for word in bangSplit]
            bangLens = [len(x) for x in bangSplit]

            if len(bangLens) - np.sum(bangLens) == lenDiffs[i]:
                newPTs = []
                j = 0
                for k in bangLens:
                    if k == 1:
                        newPTs.append(parseToken[j])
                        j = j + 1
                    if k > 1:
                        newPT = '='.join(parseToken[j:j+k])
                        newPT = re.sub(r'[^a-zA-Z]*=[^a-zA-Z]*','=', newPT)
                        newPTs.append(newPT)
                        j = j + k
                parseTokens[i] = newPTs
                i1, i2 = sent_to_graf(i, grafMap)
                parseTokens0[i1][i2] = newPTs
            else:
                chop = min(len(wordToken), len(parseToken))
                wordTokens[i] = wordToken[:chop]
                wordTokens[i] = [x + ';;' for x in wordTokens[i]]
                parseTokens[i] = parseToken[:chop]



    if VERBOSE:
        examples = random.sample(range(0, len(sentences)), min(30, len(sentences) - 2))        
        for example in examples:
            print(sentences[example])
            i1, i2 = sent_to_graf(example, grafMap)
            print(grafs[i1][i2])
            [print(x) for x in zip(wordTokens[example], parseTokens[example])]

    return [wordTokens, parseTokens]



# Parses, sorts, then saves to a pickle
def main(argv = None):
	if argv == None:
		argv = sys.argv

	name = argv[1].split('.')[0]
	text = open_file(argv[1])

	grafs = [[sentence for sentence in graf if len(sentence) > 0] for graf in split_into_sentences(text)]
	sentences = flatten_list(grafs)
	grafMap = make_grafMap(grafs)

	wordTokens0 = [[sentence.split() for sentence in graf] for graf in grafs]
	wordTokens = flatten_list(wordTokens0)

	synth = get_synth()

	print('parsing...')

	parse = []
	ii = 0

	for graf in grafs:
		if ii%1000 == 0:
			print(str(ii) + ' / ' + str(len(grafs)))
		# parse.append([synth.phonemesFromText_(sentence) for sentence in graf])
		parse.append([re.sub(r'\[\[.*\]\]', '',synth.phonemesFromText_(sentence)) for sentence in graf])
		ii += 1

	# parse = [[synth.phonemesFromText_(sentence) for sentence in graf] for graf in grafs]



	# parse = parse.replace("[[ibot1792]]", "")
	# parse = re.sub(r'\[\[ibot\d*\]\]', '', parse)
	

	# print(parse)


	# parse = re.sub(r'\[\[.*\]\]', '', parse)

	wordTokens0 = [[sentence.split() for sentence in graf] for graf in grafs]
	parseTokens0 = [[sentence.split() for sentence in graf] for graf in parse]

	wordLens0 = [[len(sentence) for sentence in graf] for graf in wordTokens0]
	parseLens0 = [[len(sentence) for sentence in graf] for graf in  parseTokens0]

	wordLarray0 = np.array([np.array(x) for x in wordLens0])
	parseLarray0 = np.array([np.array(x) for x in parseLens0])

	lenDiffs0 = list(np.subtract(wordLarray0, parseLarray0))
	lenDiffs0 = [list(x) for x in lenDiffs0]

	wordTokens = flatten_list(wordTokens0)
	parseTokens = flatten_list(parseTokens0)
	lenDiffs = flatten_list(lenDiffs0)

	count = Counter()
	countP = Counter()
	
	count.update(lenDiffs)

	print('sorting...')

	if argv[0]:
		print(count)

	for i in range(0, len(wordTokens)):
		# i is tracking each sentence 

		if i%1000 == 0:
			print(str(i) + ' / ' + str(len(wordTokens)))

		if lenDiffs[i] != 0:
			wordToken = wordTokens[i]
			wordLen = len(wordTokens[i])
			parseToken = parseTokens[i]
			parseLen = len(parseTokens[i])
			bangToken = ' ! '.join(wordToken) + ' !'
			bangParse = synth.phonemesFromText_(bangToken) + ' '
			bangSplit = re.split(r'![!,.:;\']* ', bangParse)
			bangSplit.pop()
			bangSplit = [word.split() for word in bangSplit]
			bangLens = [len(x) for x in bangSplit]

			if len(bangLens) - np.sum(bangLens) == lenDiffs[i]:
				newPTs = []
				j = 0
				for k in bangLens:
					if k == 1:
						newPTs.append(parseToken[j])
						j = j + 1
					if k > 1:
						newPT = '='.join(parseToken[j:j+k])
						newPT = re.sub(r'[^a-zA-Z]*=[^a-zA-Z]*','=', newPT)
						newPTs.append(newPT)
						j = j + k
				parseTokens[i] = newPTs
				i1, i2 = sent_to_graf(i, grafMap)
				parseTokens0[i1][i2] = newPTs
			else:
				if argv[0]:
					print(wordToken)
					print(parseToken)
					print(bangSplit)
					print('\n')
				chop = min(len(wordToken), len(parseToken))
				wordTokens[i] = wordToken[:chop]
				wordTokens[i] = [x + ';;' for x in wordTokens[i]]
				parseTokens[i] = parseToken[:chop]

	examples = random.sample(range(0, len(sentences)), min(30, len(sentences) - 2))

	for example in examples:
		print(sentences[example])
		i1, i2 = sent_to_graf(example, grafMap)
		print(grafs[i1][i2])
		[print(x) for x in zip(wordTokens[example], parseTokens[example])]

	out_dict = {'grafs':grafs, 'sentences':sentences, 'grafMap':grafMap, 'wordTokens':wordTokens, 'parseTokens':parseTokens}

	save_pickle(out_dict, 'parse_' + name)

	return out_dict


if __name__ == '__main__':
	main()
