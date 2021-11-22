
from utils.file_handling import open_pickle, save_pickle, open_file
from utils.utils import neighborhood, union_sets, make_listMap, thing_to_map, doppel_string, set_intersection
from phonetics.corpus_parse import parse_then_sort, make_grafMap, sent_to_graf

from collections import defaultdict, UserDict, UserList
from itertools import tee
import re
import os
import sys
import numpy as np

"""
vowel = re.compile('[AEIOU][AEIOUWHYX]')
nonVowel = re.compile('[^AEIOUWHYX12]')
"""

def _out(self, name = 'hotOutput'):
    fh = open('output/' + name + '.txt', 'w')
    print(str(self), file=fh)
    fh.close()

def scan_from_parse_space(parseToken):
    scan = re.sub(r'[^AEIOUWHYX12 ]', '', parseToken)
    scan = re.sub(r'(\d)[AEIOU][AEIOUWHYX]', r'\1', scan)
    scan = re.sub(r'[AEIOU][AEIOUWHYX]', '0', scan)
    return scan

def natural_meter(scanToken):
    scanToken = re.sub(' ', '', scanToken)
    term1 = scanToken.rfind('1')
    term2 = scanToken.rfind('2')
    if term2 > term1:
        # just split it 
        head = scanToken[:term1]
        tail = scanToken[term1:]
        head = re.sub('2', '1', head)
        tail = re.sub('2', '0', tail)
        meter = head + tail
    else:
        meter = re.sub('2', '1', scanToken)
    return meter

def accent_swap(vowel, conses):
    swapDict = {'r':{'AW':'AWr', 'UX':'AX', 'AE':'EY', 'AA':'AAr', 'EH':'EYAX', 'AY':'AYAX', 'IY':'IYrl', 'AO':'AOr'},
    'l':{'AY':'AYAX', 'IY':'IYrl', 'IH':'IHl', 'UW':'UWIX', 'EY':'EYAX'}, 
    'n':{'IH':'IHnm', 'AE':'AEnm'}, 'm':{'IH':'IHnm', 'AE':'AEnm'}, 'N':{'IH':'IY', 'AE':'EY'}}
    outVow = vowel
    if len(conses) > 0:
        cons = conses[0]
        if cons in ['r','l', 'm', 'n', 'N']:
            outVow = swapDict[cons].get(vowel, vowel)
    return outVow    

class Word:
    def __init__(self, wi, sentence):
        pass


# The idea is that you INIT a phrasecode
# and then you don't care about its corpus?

# not a bad idea to save the corpus ... even maybe instead of the wordTokens, parseTokens
# for tagging purposes 
class PhraseCode:
    def __init__(self, codeDict, corpus):
        self.si0 = codeDict['si0']
        self.si1 = codeDict['si1']
        self.wi0 = codeDict['wi0']
        self.wi1 = codeDict['wi1']
        self.corpus = corpus
        self.wordTokens = corpus.wordTokens
        self.parseTokens = corpus.parseTokens
        self.REPRFLAG = 'stress'

        self.header = ''
        self.footer = ''

    def __repr__(self):
        if self.REPRFLAG == 'stress':
            outString = self.stress_repr()
        elif self.REPRFLAG == 'words':
            outString = ' '.join(self.words())
        elif self.REPRFLAG == 'stripped':
            outString = ' '.join(self.words(STRIPPED= True))
        return outString

    # Is there a reason not to make it words & meter?
    # when are redundant lines helpful? 
    # just make another check 
    def __eq__(self, target):
        attributes = ['si0', 'si1', 'wi0', 'wi1', 'corpus']
        if all([self.__getattribute__(a) == target.__getattribute__(a) for a in attributes]):
            try:
                if self.meter == target.meter:
                    return True
            except:
                return True
        return False
    def __hash0__(self):
        attributes = ['si0', 'si1', 'wi0', 'wi1']
        h1 = ','.join([str(self.__getattribute__(a)) for a in attributes])
        h = hash(h1 + self.corpus.name)
        # h = hash(''.join([str(self.__getattribute__(a)) for a in attributes]))
        return h

    def __hash__(self):
        string = ' '.join(self.words(STRIPPED=True, LOWER=True)) + ''.join(self.phons()) + self.meter
        return hash(string)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __len__(self):
        try:
            return len(self.meter)
        except:
            return len(self.raw_meter())

    def stress_repr(self):
        outString = self.header
        try:
            outString += self.stress_string()
        except:
            outString += "'" + ' '.join(self.words()) + "'"
        outString += self.footer
        return outString        

    def tokens(self, token = 'wordTokens'):
        return self.corpus.__getattribute__(token)[self.si1]

    def set_header(self, string):
        self.header = string + ' '

    def append_header(self, string):
        if not hasattr(self, 'header'):
            self.header = ''
        self.header = self.header + string + ' '

    def prepend_header(self, string):
        if not hasattr(self, 'header'):
            self.header = ''
        self.header = string + ' ' + self.header

    def words(self, STRIPPED = False, SET = False, LOWER = False, HYPHEN = False):
        words = self.assemble_code(self.wordTokens)
        if HYPHEN:
            words = [x for w in words for x in re.split(r'[\-—]+', w)]
        if STRIPPED:
            words = [re.sub(r'[^a-zA-Z]', '', x) for x in words]
        if LOWER:
            words = [x.lower() for x in words]
        if SET:
            words = set(words)
        return words

    def parses(self):
        return self.assemble_code(self.parseTokens)

    #def tags(self):
    #    return self.assemble_code(self.tag_tokens())

    def scans(self):
        return self.assemble_code(self.scan_tokens())

    def syll_lengths(self):
        return [len(x) for x in self.scans()]

    def metered_scans(self):
        tail = self.meter
        outMeter = []
        for length in self.syll_lengths():
            outMeter.append(tail[:length])
            tail = tail[length:]
        return outMeter

    def scan_tokens(self):
        assert hasattr(self.corpus, 'scanTokens')
        return [t.split() for t in self.corpus.scanTokens]

    def parseString(self, STRIPPUNCT = False):
        pstring = ' '.join(self.parses())
        if STRIPPUNCT:
            pstring = re.sub(r'[^a-zA-Z= ]', '', pstring)
        return pstring
    def set_meter(self, meter):
        self.meter = meter

    def assemble_code(self, tokens):
        if self.si0 < self.si1:
            p0 = tokens[self.si0][self.wi0:]
            p1 = [x for i in range(self.si0+1,self.si1) for x in tokens[i]]
            p2 = tokens[self.si1][:self.wi1]
            outp = p0 + p1 + p2
        elif self.si0 == self.si1:
            outp = tokens[self.si1][self.wi0:self.wi1+1]
        return outp

    def stress_string(self):
        def assemble_string(words, stressCodes):
            for i, j in stressCodes[::-1]:
                words[i] = j * '°' + '′' + words[i]
            return ' '.join(words)

        assert(hasattr(self, "meter"))
        stresses = [i for i, x in enumerate(self.meter) if x == '1']
        stressCodes = []
        scan = self.scan_sentence()
        scanMap = make_listMap(scan)
        for stress in stresses:
            stressCodes.append(thing_to_map(stress, scanMap))
        return assemble_string(self.words(), stressCodes)

    # Somehow this is like 100x slower than stress_string0
    # is it ... metered_scans()?
    # is it 'in' ? 
    def stress_string_full(self, FULL = True):
        outString = ''
        d = {'0':'°', '1':'`'}
        outWords = []
        for word, wordMeter in zip(self.words(), self.metered_scans()):
            if not FULL:
                if not '1' in wordMeter:
                    outWords.append(word)
                    continue
            wordMarkup = ''.join(d[x] for x in wordMeter)
            outWords.append(wordMarkup + word)
        return ' '.join(outWords)



    #def phones(self, STRESS = False, SPACE = False, EQUALS = False):
    #    return self.phons(self, STRESS = STRESS, SPACE = SPACE, EQUALS = EQUALS)


    # THIS only handles raw stress... doesn't handle stress based on self.meter
    def phons(self, STRESS = False, SPACE = False, EQUALS = False):
        parseString = re.sub(r'[^a-zA-Z12= ]', '', self.parseString())
        if not SPACE:
            parseString = re.sub(' ', '', parseString)
        if not STRESS:
            parseString = re.sub(r'[12]', '', parseString)
        if not EQUALS:
            parseString = re.sub('=', '', parseString)
        elif STRESS:
            parseString = re.sub('2', '1', parseString)
        phons = re.split(r'([=a-zJNCDSTZ ]|\d?[AEIOU][AEOWHYX])', parseString)
        phons = list(filter(None, phons))

        return phons

    def phonss(self):
        cleaned = re.sub(r'[^a-zA-Z12=]', '', self.parseString())
        cleaned = re.sub('2', '1', cleaned)
        phons = re.split(r'([=a-zJNCDSTZ]|\d?[AEIOU][AEOWHYX])', cleaned)
        phons = list(filter(None, phons))
        return phons

    def get_terminal_sound(self, length, PERF = False, KEEPSPACE = False):
        phons = self.parseString(STRIPPUNCT = True)
        targetSylls = r'(\d?[AEIOU][AEIOUWHYX][^AEIOUWHYX12]*){%s}' % (length-1)
        terminalSyll = r'(\d?[AEIOU][AEIOUWHYX][^AEIOUWHYX12 ]*)'
        locRE = re.compile('(' + targetSylls + terminalSyll + ')$')
        rmatch = re.search(locRE, phons)
        if rmatch != None:
            rhymeSound = rmatch.group(1)
        else:
            rhymeSound = ''
        if not PERF:
            rhymeSound = re.sub(r'[^AEIOUWHYX]', '',rhymeSound)
        else:
            if not KEEPSPACE:
                rhymeSound = re.sub(' ', '', rhymeSound)
        return rhymeSound
    
    def stress_pos(self):
        assert(hasattr(self, 'meter'))
        length = len(self.meter) - self.meter.rfind('1')
        return length

    def raw_stress_pos(self):
        meter = self.rawer_meter()
        stressPos = meter.rfind('1')
        if stressPos == -1:
            stressPos = meter.rfind('2')
        if stressPos == -1:
            length = len(meter)
        else:
            length = len(meter) - stressPos
        return length

    def syllabify(self, parseString = None):
        if parseString == None:
            parseString = self.parseString(STRIPPUNCT = True)
        parseString = re.sub(r'[12]', '', parseString)
        parseWords = parseString.split()
        syllabWords = [re.split(r'([AEIOU][AEIOUWHYX])', p) for p in parseWords]
        return syllabWords

    def rhyme_sound(self, PERF = False):
        def accent_vowRhyme(rhymeSound):
            # syllabify in the sense of ['cons', 'VOW', 'cons', 'VOW', 'cons']
            # for each vow, pass and first cons it to accent_swap()
            # take the output 
            outVows = []
            syllbses = self.syllabify(rhymeSound)
            for syllbs in syllbses:
                for i in range(1,len(syllbs), 2):
                    outVows.append(accent_swap(syllbs[i], syllbs[i+1]))
            return outVows

        length = self.stress_pos()

        perfRhyme = self.get_terminal_sound(length, PERF = True, KEEPSPACE = True)
        if PERF:
            rhymeSound = re.sub(' ', '', perfRhyme)
        else:
            vows = accent_vowRhyme(perfRhyme)
            rhymeSound = ''.join(vows)
        return rhymeSound

    def allit_sound(self, PERF = False):
        parseString = " ".join(self.parseWords)
        allitString = re.sub(r'\d?[A-Z]{2}.*$', '', parseString)
        if allitString == "":
            allitString = re.sub(r'^(\d?[A-Z]{2}).*$', '\1', parseString)
        return allitString

    # would be nice to have a syllabify 
    # returns the full parseword of the 
    # would of course also be nice to rewrite get_terminal_sound to not strip
    # Right now 'whatever' and 'ever' rhyme... but I don't think there's anything you can do

    # 'unforgettable' gives 'rgEHtAXbAXl'
    # should really be 'gEHtAXbAXl'
    # probably a smart way to think about which consonants would join
    def rhyme_word(self):
        length = self.stress_pos()
        if length == len(self.meter):
            rhymeWord = self.parseString(STRIPPUNCT = True)
        else:
            length += 1
            rhymeBlop = self.get_terminal_sound(length, PERF = True, KEEPSPACE = True)
            rhymeBlop = re.sub(r'^[^AEIOUWHYX12]*[AEIOU][AEIOUWHYX]', '', rhymeBlop)
            # I do doubt it's possible for there to be a ' ' and an '='
            s = re.search(r'^([^AEIOUWHYX12]*[ =])+(.*)', rhymeBlop)
            if s != None:
                rhymeBlop = s.groups()[1]
            rhymeWord = rhymeBlop
        rhymeWord = re.sub(r' ','',rhymeWord)
        return rhymeWord

    def rhymes(self, target, PERF = True):
        return self.rhyme_sound(PERF) == target.rhyme_sound(PERF)

    # better to return list?
    # This is incredibly slow
    def term_word(self):
        length = self.stress_pos()
        pos = length - 1
        scans = self.scans()
        syllMap = make_listMap(scans[::-1])
        wi = thing_to_map(pos, syllMap)[0]
        word = self.words()[-(wi+1):]
        return ' '.join(word)

    # This is sort of what I want
    # Always STARTS AT VOWEL
    def get_sound_at_loc(self, location, length, PERF = False):
        phons = self.parseString(STRIPPUNCT = True)
        if location == None:
            syllsBefore = r''
        elif location == 0:
            syllsBefore = r''
        else:
            syllsBefore = r'^(?:[^AEIOUWHYX12]*\d?[AEIOU][AEIOUWHYX][^AEIOUWHYX12]*){%s}' % location
        targetSylls = r'(\d?[AEIOU][AEIOUWHYX][^AEIOUWHYX12]*){%s}' % (length-1)
        terminalSyll = r'(\d?[AEIOU][AEIOUWHYX][^AEIOUWHYX12 ]*)'
        locRE = re.compile(syllsBefore + '(' + targetSylls + terminalSyll + ')')
        rmatch = re.search(locRE, phons)
        if rmatch != None:
            rhymeSound = rmatch.group(1)
        else:
            rhymeSound = ''
        if not PERF:
            rhymeSound = re.sub(r'[^AEIOUWHYX]', '',rhymeSound)
        else:
            rhymeSound = re.sub(' ', '', rhymeSound)
        return rhymeSound
    def scan_sentence(self):
        parseSentence = ' '.join(self.parses())
        return scan_from_parse_space(parseSentence).split()

    def scan_sentence0(self):
        parseSentence = self.parseTokens[self.si1][:self.wi1]
        return scan_from_parse_space(parseSentence).split()

    def set_raw_meter(self):
        self.meter = self.raw_meter()
        return

    def pos_doppel(self):
        return doppel_string(self.first_poses(), self.words())
        # wish I could do self.__repr__().split
        # But I need to exclude header & footer
        # return doppel_string([p[0] for p in self.poses()], self.words())
#         pass

    # This is redundant w/ pos_doppel(), but probably better
    def doppel(self):
        print(doppel_string([' '.join(x) for x in self.poses()], self.words()))

    def phon_doppel(self):
        return doppel_string(self.parses(), self.words())

    def meter_doppel(self):
        pass

    # Could also add one for phrases... with the length
    # (this is for a whole sentence)

    # GovSets (with word?)

    def rawer_meter(self):
        scanToken = ''.join(self.scans())
        return scanToken

    def raw_meter(self):
        scanToken = ''.join(self.scans())
        meter = natural_meter(scanToken)
        return meter

    def re_search(self, targetMeter):
        # Doesn't work bc doesn't set xes right...
        def re_com_assemble(meterCom):
            comD = {'s':r'^', 'c':r' ?', 'e':r'$','1':r'[12]', '0':r'[012]', 'O':r'0', 'x':r'([012])?', 'X':r'([12])?', 'y':r'(0)?'}
            #if FULLSTOP:
            #    comD['s'] = r'(?=((^)'
            #    comD['e'] = r'($)))'
            #'0101'
            #r'^[012][12][012][12]$'
            reString = comD['s'] + ''.join([comD[syll] for syll in meterCom]) + comD['e']
            #if PLUS:
            #    target = comD['c'].join([comD[syll] for syll in meterCom]) + comD['c']
            #    reString = comD['s'] + '(' + target + ')+' + comD['e']
            reCommand = re.compile(reString)
            return reCommand

        meter = self.raw_meter()
        targetRE = re_com_assemble(targetMeter)
        #targetRE = Corpus._com_assemble(None, targetMeter, FULLSTOP = True)
        s = re.search(targetRE, meter)
        if s:
            meterCom = list(targetMeter)
            subDict = {'X':'1','x':'0', 'y':'O'}
            sregs = s.regs[1:]
            for a, _  in sregs:
                if a == -1:
                    continue
                # This would allow a () pass
                # meterCom[a:b] = [subDict[t] for t in targetMeter[a:b]]
                meterCom[a] = subDict[targetMeter[a]]
            outMeter = re.sub(r'[xXy]', '', ''.join(meterCom))
            self.set_meter(outMeter)
            return True
        else:
            return False

    # rawer_meter is very slow
    def _test_meter(self):
        conversionDict = {'1':1, '2':0.5, '0':0}
        meter = np.array([conversionDict[x] for x in self.meter])
        rawMeter = np.array([conversionDict[x] for x in self.rawer_meter()])
        diff = meter ** 2 - rawMeter ** 2
        total = sum(abs(diff))
        fraction = total / len(meter) ** 2
        return fraction

    def find_meter(self):
        # get scanTokens
        # (get the chunk)
        # push (01)s back
        # (basically can do a regex search of)
        # (001)+xx
        # xx(01)+x
        # x(10)+

        pass

    def test_w0(self, word):
        own0 = self.words()[0].lower()
        own0 = re.sub(r'[^\w]', '', own0)
        word = re.sub(r'[^\w]', '', word)
        return bool(re.match(word, own0))
        # return  own0 == word

    def test_word_re(self, reg):
        ownWord = ' '.join(self.words())
        return bool(re.search(reg, ownWord))

    def test_termWord(self, termWord):
        return self.term_word() != termWord

    # DOES this do what I want... allows an under-match
        # which is usually good
        # Have to put in my own ^ or $
    def test_pos0(self, posRE):
        own0 = self.poses()[0]
        return any([re.match(posRE, p) for p in own0])
#         return pos in own0

    def test_pos_re(self, reg):
        ownPos = ' '.join(['|'.join(p) for p in self.poses()])
        return bool(re.search(reg, ownPos))

    def test_gTag(self, tag):
        tags = self.grammarTags
        return tag in tags
        # ownPos = ' '.join(['|'.join(p) for p in self.poses()])
        # return bool(re.search(reg, ownPos))


    def test_length(self, l0, l1 = None):
        if l1 is None:
            result = len(self) == l0
        elif l1 == -1:
            result = len(self) <= l0
        elif l1 == 0:
            result = len(self) >= l0
        elif float(l1):
            if len(self) >= l0:
                result = len(self) <= l1
            else:
                result = False
        return result

    def test_word_length(self, l0, l1 = None):
        if l1 is None:
            result = len(self.words()) == l0
        elif l1 == -1:
            result = len(self.words()) <= l0
        elif l1 == 0:
            result = len(self.words()) >= l0
        elif float(l1):
            if len(self.words()) >= l0:
                result = len(self.words()) <= l1
            else:
                result = False
        return result


class Phone:
    def __init__(self, phoneme):
        if isinstance(phoneme, int):
            self.code = phoneme
            self.phone = self.code_to_phone(self.code)
        else:
            self.phone = phoneme
            self.code = self.phone_to_code(self.phone)

    def __repr__(self):
        return str(self.phone)
     
    # decode a one hot encoded string
    def dec_one_hot(encoded_seq):
        return [np.argmax(vector) for vector in encoded_seq]

    def code_to_phone(self, code):
        return self.phonephabet[code]

    def phone_to_code(self, phone):
        return self.phonephabet.index(phone)

# Probably be able to take a phrase? 

class Phonemes:
    def __init__(self, parseWords):
        self.parseWords = parseWords
        self.parseString = ' '.join(parseWords)
#         self.phons

    # Can keep 1. stresses, 2. spaces, 3. puncts
    # Different functions for each? 

    def phons_splitter(self):
        cleaned = re.sub(r'\[\[.*\]\]', '', self.parseString)
        cleaned = re.sub(r'[^a-zA-Z 12=]', '', cleaned)
        phons = re.split(r'([=a-zJNCDSTZ]|\d?[AEIOU][AEOWHYX])', cleaned)
        phons = list(filter(None, phons))
        return phons

    def phons_splitter_strict(self):
        cleaned = re.sub(r'\[\[.*\]\]', '', self.parseString)
        cleaned = re.sub(r'[^a-zA-Z]', '', cleaned)
        phons = re.split(r'([=a-zJNCDSTZ]|[AEIOU][AEOWHYX])', cleaned)
        phons = list(filter(None, phons))
        return phons

    # Dealing with them as regexes is just a worse version of putting them into a one-hot-vector
    def strict_phons(self):
        phons = self.phons_getter()
        f = filter(lambda x: x not in [' ', '='],k)
        return list(f)

    def enc_one_hot(sequence, n_unique = 40):
        encoding = list()
        if sequence == None:
            vector = [0 for _ in range(n_unique)]
            encoding.append(vector)
            print('vvv', vector, 'passed empty sequence')
            return np.array(encoding)
        for value in sequence:
            vector = [0 for _ in range(n_unique)]
    #        if value != 'q':
            vector[value] = 1
            encoding.append(vector)
        return np.array(encoding)


class Picks(list):
    def __repr__(self):
        return '\n'.join([str(x) for x in self])
    def __add__(self, target):
        return Picks(list.__add__(self, target))

    def rhymed(self, PERF = True, n = 1):
        sortedP = SortedPicks(self, PERF)
        sortedP.rh(n)
        return sortedP

    def out(self, name = 'hotOutputPicks'):
        fh = open('output/' + name + '.txt', 'w')
        print(str(self), file=fh)
        fh.close()

    def len_sort(self):
        return Picks(sorted(self, key=len))

    # Don't think I'll really use this 
    def meta_copy(self):
        copy = Picks()
        copy.stanza = getattr(self, 'stanza', None)
        copy.frame = getattr(self, 'stanza', None)
        return copy

    def red(self):
        return Picks(set(self))

    def cross_rh(self):
        rhymeWords = defaultdict(set)
        for pc in self:
            rhymeWords[pc.corpus.name].add(pc.rhyme_word())
        return rhymeWords

    def rhyme_words(self):
        rhymeWords = set([pc.rhyme_word() for pc in self])
        return rhymeWords

    def term_words(self):
        termWords = set([pc.term_word() for pc in self])
        return termWords
        # also just want a list of the words... term... 
        # how to get from ... raw parse? 
        # or code assmemble tagTokens

    def term_subset(self, excludeWord, DEL = False):
        if DEL:
            for pc in reversed(self):
                if not pc.term_word().lower() == excludeWord.lower():
                    del pc
            return
        else:
            return Picks([pc for pc in self if not pc.term_word().lower() == excludeWord.lower()])

    def cull_self(self, testFunction, *args):
        ogLen = len(self)
        for i, pc in enumerate(reversed(self)):
            i = ogLen - 1 - i
            if not testFunction(pc, *args):
                del self[i]

    def culled_inds(self, testFunction, *args):
        deleteInds = []
        for i, pc in enumerate(self):
            if not testFunction(pc, *args):
                deleteInds.append(i)
        return deleteInds

    def _w0(self, wordRE):
        self.cull_self(PhraseCode.test_w0, wordRE)

    def _w_re(self, wordRE):
        self.cull_self(PhraseCode.test_word_re, wordRE)

    def _pos0(self, posRE):
        self.cull_self(PhraseCode.test_pos0, posRE)

    def _pos_re(self, posRE):
        self.cull_self(PhraseCode.test_pos_re, posRE)

    def _len_cull(self, l0, l1 = None):
        self.cull_self(PhraseCode.test_length, l0, l1)

    def _word_len_cull(self, l0, l1 = None):
        self.cull_self(PhraseCode.test_word_length, l0, l1)        

    def de_rhy(self, pc):
        term = pc.term_word()
        self.cull_self(PhraseCode.test_termWord, pc.term_word())
        pass

    def re_search(self, targetMeter):
        return Picks([pc for pc in self if pc.re_search(targetMeter)])

    def grr(self):
        for pc in self:
            pc.header_grammar()
        self.sort(key= lambda pc: pc.header)


    def grammar_set(self):
        return set([x for pc in self for x in pc.grammarTags if pc.grammarTags])

    # NO way to allow... empty tags...
    def grammar_subset(self, tags):
        # if not hasattr(self)
        try:
            return Picks([pc for pc in self if set_intersection(pc.grammarTags, tags)])
        except:
            print("Probably passed a pc that didn't get grammar tagged")
            return 

    def header_justify(self, totalMax = 6):
        maxLen = min(totalMax, max([len(pc.header) for pc in self]))
        for pc in self:
            head = pc.header[:totalMax]
            pc.set_header(head.ljust(maxLen))

    def set_stanza(self, stanza):
        self.stanza = stanza

    def copy_self(self):
        picksCopy = Picks([x for x in self])
        if hasattr(self, 'frame'):
            picksCopy.frame = self.frame
        if hasattr(self, 'stanza'):
            picksCopy.stanza = self.stanza
        return picksCopy

    def pad(self):
#        p = self
#        pc0 = self[0]
        pc1 = self[-1]
#        for _ in range(5):
#            p = Picks(pc0) + p
        for _ in range(5):
            self.append(pc1)

    def score_meters(self, meterDict):
        # This is just going to tag each PC with a meter score ...............
        # right?
        values = {"0":0, "1":2, "2":1}
        # WOULD BE VERY EASY TO ADD 
            # Its little match feet
            # Thats the advantage of keeping the resultsDict
            # You can store the diffs, which have information by 
        # 
        # resultsDict = {}
        for targetMeter in meterDict:
            # 
            targetArray = np.array([values[x] for x in targetMeter])
            #
            matchArrays = [np.array([values[x] for x in matchMeter]) for matchMeter in meterDict[targetMeter]]
            M = np.array(matchArrays)
            diffs = (targetArray - M)
            scores = np.sum(diffs ** 2, axis=1)
            scores = 1 - (scores / len(targetMeter) ** 2)
            # resultsDict[targetMeter] = {matchMeter: score for score, matchMeter in zip(scores, meterDict[targetMeter])}
            for score, meter in zip(scores, meterDict[targetMeter]):
                for index in meterDict[targetMeter][meter]:
                    self[index].meterScore = score

    def meter_sort(self):
        self.sort(key= lambda pc: pc.meterScore)

    def meter_tag(self):
        scores = np.array([pc.meterScore for pc in self])
        oneThird = np.percentile(scores, 33)
        twoThirds = np.percentile(scores, 66)
        for pc in self:
            for tag, threshold in zip("ø*•", [oneThird, twoThirds, np.Inf]):
                if pc.meterScore < threshold:
                    pc.append_header(tag + " ")
                    break


# the same thing as an iList
# Okay so then do you also sort it???
# 

# The next thing is
    # Picks wrapper for RhymeBlocks

    # KEEP THE WHOLE RHYME BLOCK
    # If ONE pc follows this ...

    # Get rid of one word in a picks
        # One end-word

# 


while False:

    pw = PicksWrapper(picks)
    pw._update_buffer([0,1,2,3,4])
    pw.go_back()

# A wrapper for picks that knows its rhyme
# has some special methods
class RhymeBlock(Picks):
    def __init__(self):
        pass

# I should sort the dict by...
# key length... or number of matches?
while False:
    clams = SortedPicks.i_rhymed(None, s101, PERF=False)

    # Can make this into a graph
    # 




class SortedPicks(defaultdict):
    def __init__(self, picks, PERF = True):
        self.default_factory = Picks
        # Not ducktyping! 
        if isinstance(picks, Picks):
            if PERF:
                self.perf_rhymed(picks)
            else:
                self.vow_rhymed(picks)
        # Should I pass something to allow this to sort it?
        elif isinstance(picks, defaultdict):
            for key in list(picks):
                self.__setitem__(key, picks[key])

    def __repr__(self):
        return '\n'.join(['\n' + key + '\n' + str(value) for key, value in self.items()])

    def __len__(self):
        return np.sum([len(value) for value in self.values()])



# SORT BY length of rhyme
# and then by... number of elements? 

    def vow_rhymed(self, picks):
        [self[pc.rhyme_sound(PERF = False)].append(pc) for pc in picks]
    def perf_rhymed(self, picks):
        [self[pc.rhyme_sound(PERF = True)].append(pc) for pc in picks]
    def i_rhymed(self, picks, PERF = False):
        dicto = defaultdict(list)
        for i, pc in enumerate(picks):
            rhyme = pc.rhyme_sound(PERF=PERF)
            dicto[rhyme].append(i)
        return dicto
    def cull_keys(self, test_function, argsList, NOT = False):
        for key in list(self):
            test = test_function(self, self.__getitem__(key), *argsList)
            if NOT:
                test = not test
            if test:
                self.__delitem__(key)

    # deletes any individual pick that doesn't pass a test
    def cull_picks(self, args):
        for key in list(self):
    #        self.__setitem__(key) = self.__getitem__(key).__cull_picks(*args)
            pass
    def bll(self, n = 2):
        # self.cull_keys(len, [])
        for key in list(self):
            if len(self.__getitem__(key)) < n:
                self.__delitem__(key)


    def red(self):
        for value in self.values():
            value = value.red()

    def rh(self, n = 2):
        for key in list(self):
            # how do you pass this? 
            if len(self.__getitem__(key).rhyme_words()) < n:
                self.__delitem__(key)

# Just need the number to be over two for both of them 
# Need to know the number of corpora at play here

    def lnrh(self, n = 1, m = 1):
        for key in list(self):
            pass
            # Get all of the lens
            # try to assert that there are two similar lens that have different rhymewords

    def ccrh(self, n = 1, m = None):
        length = self.corp_tag()
        if m != None:
            length = m
        self.cross_corp_rh(m = length, n = n)

    def cross_corp_rh(self, m = 2, n = 1):
        for key in list(self):
            sets = self.__getitem__(key).cross_rh()
            sets = list(sets.values())
            if len(sets) >= m:
                lens = [len(s) for s in sets]
                if all(l > n for l in lens):
                    continue
                combine = union_sets(sets)
                if len(combine) - max(lens) >= n:
                    continue
            self.__delitem__(key)

    def corp_tag(self):
        ks = ['+', '-', '$', '#', '%', '&', '*']
        corps = []
        for key in list(self):
            for pc in self.__getitem__(key):
                if pc.corpus not in corps:
                    corps.append(pc.corpus)
                ii = corps.index(pc.corpus) % len(ks)    
                head = ks[ii]
                pc.append_header(head)
        return len(corps)

    def out(self, name = 'hotOutputSorted'):
        fh = open('output/' + name + '.txt', 'w')
        print(str(self), file=fh)
        fh.close()

    def len_sort(self):
        for key in list(self):
            picks = self.__getitem__(key)
            self.__setitem__(key, picks.len_sort())

    def sff(self):
        for key in list(self):
            picks = self.__getitem__(key)
            picks.sff()

    def header_justify(self):
        for key in list(self):
            self.__getitem__(key).header_justify()

    def melt(self):
        # outPicks = Picks([x for key in list(self) for x in self[key]])
        outPicks = Picks()
        for key in list(self):
            outPicks += self[key]
        return outPicks

    def set_stanza(self, stanza):
        self.stanza = stanza

    # Redesign to throw to a series of tabs?
    def throw(self):
        assert(len(self)) > 0
        for rhyme in list(self):
            pick = self[rhyme]
            pick.set_stanza(self.stanza)
            pick.throw(prename = rhyme + ' ')
            # pc = pick[0]
            # rootName = rhyme + ' ' + pc.corpus.name + ' ' + pc.meter
            # root = tk.Tk()
            # root.title(rootName)
            # self.frame = tt.FramePicks(root, self)
            # self.frame.pack(fill="both", expand=True)



class Corpus:
    def __init__(self, name):
        self.name = name
        self.wordTokens, self.parseTokens = self.try_open_parse()
        self.get_scanTokens()
        self.grammarTokens = [[] for i in range(len(self))]

    def __len__(self):
        return len(self.wordTokens)

    def __repr__(self):
        wordLen = self.word_len()
        lineLen = len(self)
        lineMeans = np.mean([len(sentence) for sentence in self.wordTokens])
        wordMeans = np.mean([len(word) for sentence in self.wordTokens for word in sentence])
        phonDensity = np.mean([len(pword)-1 for psentence in self.parseTokens for pword in psentence]) / wordMeans
        outString = 'Corpus from %s.txt\n%d sentences\n%d words\nMean line: %.1f words\nMean word: %.1f chars\nPhonDensity %.2f' % (self.name, lineLen, wordLen, lineMeans, wordMeans, phonDensity)
        return outString

    def word_len(self):
        return np.sum([len(sentence) for sentence in self.wordTokens])


    def parseSentence(self, si):
        return ' '.join(self.parseTokens[si])

    def try_open_parse(self):
        try:
            inDict = open_pickle('parse' + '_' + self.name)
            wordTokens = inDict['wordTokens']
            parseTokens = inDict['parseTokens']
        except:
            print('Have to read that input file')
            text = open_file(self.name)
            wordTokens, parseTokens = parse_then_sort(text)
            save_pickle({'wordTokens':wordTokens, 'parseTokens':parseTokens}, 'parse' + '_' + self.name)
        return wordTokens, parseTokens
            # The new thing would be to save the pickle with the two dicts also
            # And then just write all of the handler methods to translate them 

    def scan_sentence_index(self, si):
        scan = re.sub(r'[^AEIOUWHYX12 ]', '', self.parseSentence(si))
        scan = re.sub(r'(\d)[AEIOU][AEIOUWHYX]', r'\1', scan)
        scan = re.sub(r'[AEIOU][AEIOUWHYX]', '0', scan)
        return scan

    def get_scanTokens(self):
        self.scanTokens = []
        self.scanMaps = []
        for si in range(len(self.parseTokens)):
            scan = self.scan_sentence_index(si)
            scanList = scan.split(' ')
            scanMap = make_grafMap(scanList, pad=1)
            self.scanTokens.append(scan)
            self.scanMaps.append(scanMap)
        return

    def esearch(self, meterCom, REV):
        return self.regex_meter_search(meterCom, REV, EITHER = True)
        pass

    def search(self, meterCom, REV = True):
        return self.regex_meter_search(meterCom, REV, FULLSTOP = False)

    # Grammar search: meter search Corpus while also grammar tagging
    def gsearch(self, meterCom, REV = True, FULLSTOP = False, CULL = 0, SCORE_METER = True):
        return self.regex_meter_search(meterCom, REV, FULLSTOP, TAG = True, CULL = CULL, SCORE_METER = SCORE_METER)

    # Full search: meter search Corpus restricted to full sentences
    def fsearch(self, meterCom, REV = True):
        return self.regex_meter_search(meterCom, REV, FULLSTOP = True)

    def ssearch(self, meterCom, REV = True):
    	return self.regex_meter_search(meterCom, REV, STARTSTOP = True)

    # Search for a small foot like '(001)+', returns 001, 001001, 001001001, etc.
    def search_plus(self, meterCom, REV = True, FULLSTOP = False, STARTSTOP= False):
        return self.regex_meter_search(meterCom, REV, FULLSTOP = FULLSTOP, STARTSTOP = STARTSTOP, PLUS = True)

    def ssearch_plus(self, meterCom, REV = True):
        return self.regex_meter_search(meterCom, REV, STARTSTOP = True, PLUS = True)

    # This is for quickly getting a picks that doesn't care much about meter from an input that is just a list of words
    # Maxes out at like 20 syllables or something
        # This is for reading in lines that are all one word or phrase long
    # Which in real life ... a lot of sentences are longer than
    def aaa(self):
        return self.fsearch('xxxxxxxxxxxxxxxxxXxXxXx1xxxxxxxxxxxxxxxxxx')    

    # Returns a picks for every sentence, meter tagged with its raw/natural meter 
    def natural_lines(self, RAWER = False):
        if not hasattr(self, 'scanTokens'):
            self.get_scanTokens()
        picks = Picks()
        # This should big time be a function ... on an si I guess 
        for si, scanToken in enumerate(self.scanTokens):
            if RAWER:
                meter = re.sub(' ', '', scanToken)
            else:
                meter = natural_meter(scanToken)
            pcDict = {'si0':si, 'si1':si, 'wi0':0, 'wi1':len(scanToken)-1}
            pc = PhraseCode(pcDict, self)
            pc.set_meter(meter)
            picks.append(pc)
        return picks


# It would be smart to make a Sentence object and rewrite search as a function of that

    # wild x's doesn't really work for multiple different x's in a row:
        # 'XxXx' ... it would allow ['X', 'XX', 'xx', 'XxX'], etc.
    # Adding a lot of parens might help

    def regex_meter_search(self, meterCom, REV = True, FULLSTOP = False, STARTSTOP = False, PLUS = False, TAG = False, CULL = 0, MAX = None, SCORE_METER = False, EITHER = False):
        def com_assemble(meterCom, FULLSTOP = False, STARTSTOP= False, PLUS = False, REV = True):
            s = r'(?=((^| )'
            ss = r'(?=((^)'
            c = r' ?'
            V = r'[12]'
            v = r'0'
            w = r'[012]'
            e = r'( |$)))'
            ee = r'($)))'
            x = r'(' + w + r')?'
            X = r'(' + V + r')?'
            y = r'(' + v + r')?'
            comD = {'s':r'(?=((^| )', 'c':r' ?', 'e':r'( |$)))','1':r'[12]', '0':r'[012]', 'O':r'0', 'x':r'(' + w + r')?', 'X':r'(' + V + r')?', 'y':r'(' + v + r')?'}
            # if REV:
            #     meterCom = meterCom[::-1]
            if FULLSTOP:
                comD['s'] = ss
                comD['e'] = ee
            if STARTSTOP:
                if REV:
                    comD['e'] = ee
                else:
                    comD['s'] = ss
            # comD = {'s':s, 'c':c, 'e':e ,'1':V, '0':w, 'O':v, 'x':r'(' + w + r')?', 'X':r'(' + V + r')?', 'y':r'(' + v + r')?'}
            reString = comD['s'] + comD['c'].join([comD[syll] for syll in meterCom]) + comD['e']
            # beef it up to add xs after
            if PLUS:
                target = comD['c'].join([comD[syll] for syll in meterCom]) + comD['c']
                reString = comD['s'] + '(' + target + ')+' + comD['e']
            reCommand = re.compile(reString)
            return reCommand

        # Do this just for one at a time? 
        def com_assemble_plus(meterCom, FULLSTOP = False):
            s = r'(?=((^| )'
            ss = r'(?=((^)'
            c = r' ?'
            V = r'[12]'
            v = r'0'
            w = r'[012]'
            e = r'( |$)))'
            ee = r'($)))'
            x = r'(' + w + r')?'
            X = r'(' + V + r')?'
            y = r'(' + v + r')?'
            comD = {'s':r'(?=((^| )', 'c':r' ?', 'e':r'( |$)))','1':r'[12]', '0':r'[012]', 'O':r'0', 'x':r'([012])?', 'X':r'([12])?', 'y':r'([12])?'}
            if FULLSTOP:
                comD['s'] = ss
                comD['e'] = ee
            #
            #r'(?=((^| )([12] ?[012] ?[012] ?)+( |$)))'
            #r'(?=((^| )[12] ?[012] ?[012]( |$)))'
            #
            target = comD['c'].join([comD[syll] for syll in meterCom]) + comD['c']
            reString = comD['s'] + '(' + target + ')+' + comD['e']
            reCommand = re.compile(reString)
            return reCommand

        def xs_to_outMeter(rgroup, meterCom, xIndices):
            assert(len(xIndices) == len(rgroup))
            subDict = {'X':'1','x':'0', 'y':'O'}
            meter = list(meterCom)
            for i, r in enumerate(rgroup[::-1]):
                ii = xIndices[len(xIndices)-i-1]
                if r == None:
                    del meter[ii]
                else:
                    meter[ii] = subDict[meter[ii]]
            return ''.join(meter)

        def xs_to_outMeters(rgroups, meterCom):
            xIndices = [i for i, x in enumerate(meterCom) if x in ['X', 'x', 'y']]
            outMeters = [xs_to_outMeter(rgroup, meterCom, xIndices) for rgroup in rgroups]
            return outMeters

        def forward_search(self, meterCom, FULLSTOP = False, STARTSTOP = False):
            reCommand = com_assemble(meterCom, FULLSTOP = FULLSTOP, STARTSTOP = STARTSTOP, REV = False)
            picks = Picks()
            for si in range(len(self.parseTokens)):
                iterMatches, iterXes = tee(re.finditer(reCommand, self.scanTokens[si]))
                # iterMatches, iterXes = tee(re.finditer(reCommand, scanToken))
                matchIndices = [[i.start(1), i.end(1)] for i in iterMatches]
                scanIndices = [[sent_to_graf(ai, self.scanMaps[si])[0], sent_to_graf(zi, self.scanMaps[si])[0]] for ai, zi in matchIndices]
                rgroups = [i.groups()[2:-1] for i in iterXes]
                # Get an assertion error here bc 
                meters = xs_to_outMeters(rgroups, meterCom)
                for i, (ai, zi) in enumerate(matchIndices):
                    wi0 = sent_to_graf(ai+1, self.scanMaps[si])[0]
                    wi1 = sent_to_graf(zi-1, self.scanMaps[si])[0]
                    pcDict = {'si0':si, 'si1':si, 'wi0':wi0, 'wi1':wi1}
                    pc = PhraseCode(pcDict, self)
                    pc.set_meter(meters[i])
                    picks.append(pc)
            return picks

        def reverse_search(self, meterCom, FULLSTOP = False, STARTSTOP=False):
            revCommand = com_assemble(meterCom[::-1], FULLSTOP = FULLSTOP, STARTSTOP = STARTSTOP, REV = True)
            picks = Picks()
            # Make this a single function.... so that we can use it
            for si in range(len(self.parseTokens)):
                stLen = len(self.scanTokens[si])
                revMatches, revXes = tee(re.finditer(revCommand, self.scanTokens[si][::-1]))
                revIndices = [[stLen - i.end(1), stLen - i.start(1)] for i in revMatches]
                fwdIndices = revIndices[::-1]
                # [self.scanTokens[si][ai:zi] for ai, zi in revIndices]
                revGroups = [i.groups()[2:-1][::-1] for i in revXes][::-1]
                meters = xs_to_outMeters(revGroups, meterCom)
                for i, (ai, zi) in enumerate(fwdIndices):
                    wi0 = sent_to_graf(ai+1, self.scanMaps[si])[0]
                    wi1 = sent_to_graf(zi-1, self.scanMaps[si])[0]
                    pcDict = {'si0':si, 'si1':si, 'wi0':wi0, 'wi1':wi1}
                    pc = PhraseCode(pcDict, self)
                    pc.set_meter(meters[i])
                    picks.append(pc)
            return picks

        # The reverse version of this super doesn't work
        def either_search(self, meterCom, FULLSTOP = False, REV = False):
            if REV:
                searchCom = meterCom[::-1]
            else:
                searchCom = meterCom
            reCommand = com_assemble(searchCom, FULLSTOP = FULLSTOP, STARTSTOP = STARTSTOP, REV = False)
            picks = Picks()
            for si in range(len(self.parseTokens)):
                stLen = len(self.scanTokens[si])
                if REV:
                    scanToken = self.scanTokens[si][::-1]
                else:
                    scanToken = self.scanTokens[si]
                iterMatches, iterXes = tee(re.finditer(reCommand, scanToken))
                if REV:
                    revIndices = [[stLen - i.end(1), stLen - i.start(1)] for i in iterMatches] # revIndices
                    matchIndices = revIndices[::-1]
                    rgroups = [i.groups()[2:-1][::-1] for i in iterXes][::-1]
                else:
                    matchIndices = [[i.start(1), i.end(1)] for i in iterMatches]
                    rgroups = [i.groups()[2:-1] for i in iterXes]
                meters = xs_to_outMeters(rgroups, meterCom)
                for i, (ai, zi) in enumerate(matchIndices):
                    wi0 = sent_to_graf(ai+1, self.scanMaps[si])[0]
                    wi1 = sent_to_graf(zi-1, self.scanMaps[si])[0]
                    pcDict = {'si0':si, 'si1':si, 'wi0':wi0, 'wi1':wi1}
                    pc = PhraseCode(pcDict, self)
                    pc.set_meter(meters[i])
                    picks.append(pc)
            return picks
        
        def pluses_to_outMeters(meterTarget, matchStrings):
            l1 = len(meterTarget)
            l2 = [len(s) for s in matchStrings]
            rp = [(l / l1) for l in l2]
            meters = [meterTarget * int(r) for r in rp]
            return meters

        def plus_search(self, meterCom, FULLSTOP = False, STARTSTOP = False, REV = False):
            #command = com_assemble_plus(meterCom[::-1], FULLSTOP)            
            #matchIndices = [[i.start(1), i.end(1)] for i in iterMatches]
            #scanIndices = [[sent_to_graf(ai, self.scanMaps[si])[0], sent_to_graf(zi, self.scanMaps[si])[0]] for ai, zi in matchIndices]
            # could detangle shorter scanIndices?

            revCommand = com_assemble(meterCom[::-1], FULLSTOP=FULLSTOP, STARTSTOP=STARTSTOP, REV=REV, PLUS = True)
            picks = Picks()
            for si in range(len(self.parseTokens)):
                stLen = len(self.scanTokens[si])
                revMatches, revMatchStrings = tee(re.finditer(revCommand, self.scanTokens[si][::-1]))
                revIndices = [[stLen - i.end(1), stLen - i.start(1)] for i in revMatches]
                fwdIndices = revIndices[::-1]

                revGroups = [i.groups()[0] for i in revMatchStrings]
                fwdStrings = [s.replace(" ", "") for s in revGroups[::-1]]
                meters = pluses_to_outMeters(meterCom, fwdStrings)

                # [self.scanTokens[si][ai:zi] for ai, zi in revIndices]
                # revGroups =     [::-1]

                # pluses_to_outMeters
                # match len of thing to a number of 001s 
                # meters = xs_to_outMeters(revGroups, meterCom)
                for i, (ai, zi) in enumerate(fwdIndices):
                    wi0 = sent_to_graf(ai+1, self.scanMaps[si])[0]
                    wi1 = sent_to_graf(zi-1, self.scanMaps[si])[0]
                    pcDict = {'si0':si, 'si1':si, 'wi0':wi0, 'wi1':wi1}
                    pc = PhraseCode(pcDict, self)
                    pc.set_meter(meters[i])
                    picks.append(pc)
            return picks

        if not hasattr(self, 'scanTokens'):
            self.get_scanTokens()

        if EITHER:
            search = either_search(self, meterCom, REV)
        # elif TAG:
        #     search = reverse_search_stanTag(self, meterCom, FULLSTOP, CULL, SCORE_METER = SCORE_METER)
        elif PLUS:
            search = plus_search(self, meterCom, FULLSTOP=FULLSTOP, STARTSTOP=STARTSTOP, REV=REV)
        elif REV:
            search = reverse_search(self, meterCom, FULLSTOP=FULLSTOP, STARTSTOP=STARTSTOP)
        else:
            search = forward_search(self, meterCom, FULLSTOP=FULLSTOP, STARTSTOP=STARTSTOP)

        return search

        # Want to skip double searches if there aren't xes in it
        # can worry about that later

        # If ever there's something where you USE the last x
        # you could be skipping


    # This will be useful for writing multi
    # regex_meter_search() but without relying on having preprocessed the parseTokens
    def regex_meter_search_raw(self):
        s = r'(?=((^| )'
        c = r'[^AEIOU12]*'
        V = r'[12][AEIOU][AEIOUWHYX]'
        v = r'[AEIOU][AEIOUWHYX]'
        w = r'[12]?[AEIOU][AEIOUWHYX]'
        ce = r'[^AEIOU12 ]*( |$)))'

        x = r'(' + w + r')?'
        X = r'(' + V + r')?'
        y = r'(' + v + r')?'
        
        r101 = re.compile(s + c + V + c + w + c + V + ce)
        
        matches = re.finditer(r101, k)
        matchIndices = [[i.start(1), i.end(1)] for i in matches]

        return

    def cross_search(self, corp, meterCom, REV = True, FULLSTOP = True):
        picks = Picks()
        for c in [self, corp]:
            p = c.regex_meter_search(meterCom, REV, FULLSTOP)
            print(c.name, len(p))
            picks += p
        sort = picks.sorted()
        sort.red()
        sort.ccrh()
        return sort

    def flender_scans(self):
        def render_scans(i):
            return [(a, b) for a, b in zip(self.wordTokens[i], self.scanTokens[i].split(' '))]
        zip_scans = [f"{b}_ {a}" for i in range(len(self.wordTokens)) for a, b in render_scans(i)]
        return zip_scans
    
    def flender_parses(self):
        def render_parses(i):
            return [(a, b) for a, b in zip(self.wordTokens[i], self.parseTokens[i])]
        zip_parses = [f"{b} {a.strip(string.punctuation)}_" for i in range(len(self.wordTokens)) for b, a in render_parses(i)]
        return zip_parses


while False:
    corp
    sentDict = defaultdict(Picks)
    # picks = []
    # pcDicts = []
    wordTokens = corp.wordTokens
    for si, wt in enumerate(wordTokens):
        l = len(wt)
        for i, j in zip(range(l), range(1,l)):
            pc1 = PhraseCode({'si0':si, 'si1':si, 'wi0':i, 'wi1':i}, corp)
            pc2 = PhraseCode({'si0':si, 'si1':si, 'wi0':i, 'wi1':j}, corp)
            if not pc1.words(STRIPPED = True, SET = True, LOWER = True, HYPHEN = True).intersection(sstops):
                sentDict[si].append(pc1)
            if not pc2.words(STRIPPED = True, SET = True, LOWER = True, HYPHEN = True).intersection(sstops):
                sentDict[si].append(pc2)
            # sentDict[si].append(pc2)
        pc3 = PhraseCode({'si0':si, 'si1':si, 'wi0':l-1, 'wi1':l-1}, corp)
        if not pc3.words(STRIPPED = True, SET = True, LOWER = True, HYPHEN = True).intersection(sstops):
            sentDict[si].append(pc3)
            # pcDicts.append({'si0':si, 'si1':si, 'wi0':i, 'wi1':j})

    crossDict = defaultdict(list)

    sstops = set(stopWords)

    for sentence in sentDict.keys():
        pm = PicksMat(sentDict[sentence], STRESS=True)
        # try:
        #     pm.S = pm.S * D
        # except:
        #     pass
        pm.neighbor_self(threshold=0.04)
        pm.cross_self()
        for (a, b), c in pm.crosses:
            # if pm[a].pc.words(STRIPPED = True, SET = True, LOWER = True, HYPHEN = True).intersection(sstops):
            #     continue
            # if pm[b].pc.words(STRIPPED = True, SET = True, LOWER = True, HYPHEN = True).intersection(sstops):
            #     continue
            crossDict[sentence].append((pm[a], pm[b], c))
    
    overlapsList = []

    for key in crossDict:
        crosses = crossDict[key]


    # Make a new PM of common overlaps
    # So you can detect an uncommon overlap
    #   OR the thing that would really help me here is 
    # ((or just common ... ))

    # USE THE GOOGLE FREQUENCY DICT
    # EXCLUDE COMMON THINGS

    # EXCLUDE ANYTHING THAT IS JUST A STOPWORD

    # The regex is
    # SEND all of the Xes 


# SLOWER for long numbers
 # than natural ors I bet
 # if it's.... all one.. character.... don't need this

def xes_to_reg1(xes):
    if xes == None:
        return r''
    c = r' ?'
    xomDict = {'x':r'[012]', 'X':r'[12]', 'y':r'0'}
    reList = [xomDict[x] for x in reversed(list(xes))]
    reList = [r'']
    for x in reversed(list(xes)):
        r = xomDict[x]
        # rr = r + reList[-1]
        rr = r
        if reList[-1]:
            rr += c
        rr += reList[-1]
        reList.append(rr)
    # NEED TO ALSO ALLOW r' ?' ... in that join
    outRE = r'(' + r'|'.join(reList[1:]) + r')?'
    return outRE


# search = 'x1001'
# xes_to_reg1('XxXx01')
# phil = Corpus('phillippe')
# picks = phil.search(search)

# search = 'xx1001001x'
# rural = Corpus('rural_short')
# picks = rural.search(search)
# rural.try_open_stanford()
# pc = picks[0]
# print(pc)
# picks.sff()
# pc.check_stanPhrase2()



#alice = Corpus('carroll-alice')
#picks = alice.fsearch(search)
#alice.try_open_stanford()
#picks.sff()

# Really needs the output to be zipped 


while False:

# try:
	napoleon = Corpus('napoleonLines')
	napoleon.get_scanTokens()
	napoleon.try_open_stanford()
	
	picks = napoleon.gsearch('001001')
	picks.grr()
	pc0 = picks[27]
	pc2 = picks[50]
	pc3 = picks[100]



if __name__ == '__main__':
    # main()
    napoleon = Corpus('napoleonLines')
    napoleon.get_scanTokens()
    picks = napoleon.regex_meter_search('XxXxXx1001xx', True)

    print(picks)

    sys.exit()



