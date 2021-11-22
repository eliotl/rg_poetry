from meter_search.corpusclass import SortedPicks, Corpus


class GrammarMixin:
    def header_grammar(self):
        gtags = self.grammarTags
        if gtags == []:
            gtags = self.govTag
        outTag = self.clause_tag_tags() + ' '.join(sorted(gtags))
        self.set_header(outTag)

    def gs(self):
        return self.tokens('grammarTokens')

    def poses(self):
        # gsPoses = [x[0] for x in self.gs().poses()]
        gsPoses = self.gs().poses()
        # These are all lists but I kind of wish they were tuples
        poses = gsPoses[self.wi0:self.wi1+1]
        return poses

    def first_poses(self):
        return [p[0] for p in self.poses()]

    def clause_tag_tags(self, tags = None):
        if tags == None:
            tags = self.grammarTags

        clauseSet = set([stanfordTagsInverse.get(tag, 'other') for tag in tags])
        if 'sentence' in clauseSet:
            t = '~'
        elif 'clauses' in clauseSet:
            t = '@'
        elif 'phrases' in clauseSet:
            t = '='
        # THIS IS NEVER HAPPENING 
        elif 'words+' in clauseSet:
            t = '+'
        else:
            t = ''
        return t    

# Grammatical information for a single word within a phrase
class GrammarWord:
    def __init__(self, si, wi, corpus):
        self.index = wi
        self.si = si
        self.corpus = corpus
        self.selfTags = []
        self.longTags = defaultdict(list)

    def __repr__(self):
        return str(self.selfTags)

    def add_tag(self, tag, indices):
        wi1 = indices[1]
        if self.index == wi1:
            self.selfTags.append(tag)
        elif self.index < wi1:
            # These should be sorted by wi...
            self.longTags[wi1].append(tag)

    def get_pos(self):
        posTag = [t for t in self.selfTags if t in stanfordTags['words+']]
        if posTag == []:
            posTag = [t for t in self.selfTags if t not in stanfordTags['phrases'] + stanfordTags['clauses']]
        if posTag == []:
            try:
                posTag = [self.selfTags[-1]]
                print('pos not in list', str(posTag))
            except:
                # UNABLE TO GIVE A CHAR TAG 
                # print(self.corpus.wordTokens[self.si])
                pass
        return posTag

    def get_tags(self, wi1):
        outTags = []
        if self.index == wi1:
            outTags = self.selfTags
        else:
            outTags = self.longTags[wi1]
        return outTags



# A list of GrammarWords
# A full sentence 
class GrammarSentence(list):
    def __init__(self, corpus, si, EMPTY = False):
        self.corpus = corpus
        self.si = si
        self.empty_init()
        self.wordToken = self.corpus.wordTokens[si]
        self.stanToken = self.corpus.stanTokens[si]
        if self.stanToken == False:
            print('false token', str(si))
            return

        if not EMPTY:
            self.tags_from_oToken()
            self.govs_map()

    def tags_from_oToken(self):
        o = self.stanToken
        wordToken = self.wordToken
        try:
            parseTree = o['parse']
        except:
            print(self.si, 'bad stanToken')
            return
        annoTokens = o['tokens']
        tokeMap = disambiguate_tokens(annoTokens, wordToken)
        self.tokeMap = tokeMap
        if isinstance(tokeMap, bool):
            print('bad tokemap', str(wordToken))
            return

        t = Tree.fromstring(parseTree)
        subs = [x for x in t.subtrees()]
        s2 = [[x.label(), len(x.leaves())] for x in subs]
        s3 = [[x.label(), len([y for y in x.subtrees()])] for x in subs]

        runningI = 0
        for i, (label, b) in enumerate(s2):
            i0, i1 = runningI, runningI+b-1
            # QUESTION HERE 
            # what to do for multi sentence PhraseCodes? Just don't take them?
            # if fetch_mapItem()[1] != 0, do we not want to include this? it's punctuation or an 's?
            wi0 = fetch_mapItem(i0, tokeMap)[0]
            wi1 = fetch_mapItem(i1, tokeMap)[0]

            self.__getitem__(wi0).add_tag(label, (wi0, wi1))
            if s3[i][1] == 1:
                runningI += 1

    def govs_map(self):
        if isinstance(self.stanToken, bool):
            print('bad stanToken', str(self.si))
            self.depMap = []
            return
        deps = self.stanToken['basicDependencies']
        depMap = {d['dependent']:d['governor'] for d in deps}
        self.depMap = depMap

    def check_phrase_govs(self, tis):
        if self.stanToken == False:
            return []
        ti0, ti1 = tis
        ts = [i+1 for i in range(ti0, ti1+1)]
        govSets = [set(get_govs(ti, self.depMap)) for ti in ts]
        govSet = set.intersection(*govSets)
        if len(govSet) > 0:
            govMatches = [x for x in govSet if x in ts]
            if any(govMatches):
                return govMatches
        return []

    # This should mega be a pc.function
    def tag_phrase(self, pc):
        gw = self.__getitem__(pc.wi0)
        tags = gw.get_tags(pc.wi1)
        pc.grammarTags = tags
        govIndex = self.check_phrase_govs((pc.wi0, pc.wi1))
        if len(govIndex) > 0:
            if len(govIndex) > 1:
                print('long gov', str(si), str(govIndex))

            govIndex = govIndex[-1]
            try:
                gov = self.stanToken['tokens'][govIndex]
            except:
                pc.govTag = []
                return
            pos = gov['pos']
            govTag = ['"' + pos]
        else:
            govTag = []
        pc.govTag = govTag

    def empty_init(self):
        self.depMap = []
        length = len(self.corpus.wordTokens[self.si])
        for wi in range(length):
            self.append(GrammarWord(self.si, wi, self.corpus))

    def poses(self):
        return [gt.get_pos() for gt in self]
        # Either get ... 

class GrammarSentencePartial(GrammarSentence):
    def __init__(self, wordToken, stanToken, EMPTY=False):
        self.corpus = corpus
        self.wordToken = wordToken
        self.stanToken = stanToken
        # self.si = si
        self.empty_init()

    def empty_init(self):
        self.depMap = []
        length = len(self.wordToken)
        for wi in range(length):
            self.append(GrammarWord(self.si, wi, self.corpus))


class GrammarSortedPicks(SortedPicks):
        def reverse_search_stanTag(self, meterCom, FULLSTOP = False, CULL = 0, SCORE_METER = False):
            revCommand = com_assemble(meterCom[::-1], FULLSTOP = FULLSTOP, STARTSTOP = STARTSTOP, REV = True)
            picks = Picks()

            if SCORE_METER:
                meterScores = defaultdict(lambda:defaultdict(list))
            # targetMeter : rawMeter : index

            if not hasattr(self,'stanTokens'):
                self.try_open_stanford()

            for si, scanToken in enumerate(self.scanTokens):
            # for si, (scanToken, stanToken) in enumerate(zip(self.scanTokens, self.parseTokens)):
            # for si in range(len(self.parseTokens)):
                stLen = len(scanToken)
                revMatches, revXes = tee(re.finditer(revCommand, scanToken[::-1]))
                revIndices = [[stLen - i.end(1), stLen - i.start(1)] for i in revMatches]
                fwdIndices = revIndices[::-1]
                revGroups = [i.groups()[2:-1][::-1] for i in revXes][::-1]
                meters = xs_to_outMeters(revGroups, meterCom)

                if len(meters) == 0:
                    continue

                try:
                    gs = self.grammarTokens[si]
                    if gs == []:
                        gs = GrammarSentence(self, si)
                        self.grammarTokens[si] = gs
                    # This doesn't recheck if its empty 
                except:
                    gs = GrammarSentence(self, si)
                    self.grammarTokens[si] = gs

                for i, (ai, zi) in enumerate(fwdIndices):
                    wi0 = sent_to_graf(ai+1, self.scanMaps[si])[0]
                    wi1 = sent_to_graf(zi-1, self.scanMaps[si])[0]
                    pcDict = {'si0':si, 'si1':si, 'wi0':wi0, 'wi1':wi1}
                    pc = PhraseCode(pcDict, self)
                    pc.set_meter(meters[i])
                    gs.tag_phrase(pc)

                    if CULL:
                        if pc.grammarTags == []:
                            if CULL == 2:
                                continue
                            if pc.govTag == []:
                                continue

                    if SCORE_METER:
                        rawMeter = re.sub(" ", "", scanToken[ai:zi])
                        meter = meters[i]
                        meterScores[meter][rawMeter].append(len(picks))

                    picks.append(pc)

            if SCORE_METER:
                picks.score_meters(meterScores)

            return picks        


class GrammarCorpus(Corpus):
    def try_open_stanford(self):
        def _chew_stanford(wordTokens):
            outList = []
            for i, t in enumerate(wordTokens):
                if i % 500 == 0:
                    print(str(i), '/', str(len(wordTokens)))
                if ';;' in ' '.join(t):
                    outList.append(False)
                else:
                    outList.append(output_for_sentence(' '.join(t)))
            return outList

        def _chew_stanford_more(wordTokens):
            ['index', 'parse', 'basicDependencies', 'enhancedDependencies', 'enhancedPlusPlusDependencies', 'tokens']
            for t in wordTokens:
                wordToken = ' '.join(t)
                o = output_for_sentence(wordToken)
            return [output_for_sentence(' '.join(t)) for t in wordTokens]
        try:
            inList = open_pickle('stan_' + self.name)
        except:
            print('Have to chew that stanford')
            from stanfordparse import output_for_sentence
            inList = _chew_stanford(self.wordTokens)
            save_pickle(inList, 'stan_' + self.name)
        self.stanTokens = inList
        print('okay')
        return
    def _mini_chew_stanford(self, wordTokens = None):
        if wordTokens is None:
            wordTokens = self.wordTokens
        i = 0
        while i < len(wordTokens):
            token = " ".join(wordTokens[i])
            if ";;" in token:
                result = False
            else:
                result = output_for_sentence(token)
            i += 1
            yield result
    def _mini_chew_stanford_random(self, wordTokens = None, excludeList = []):
        if wordTokens is None:
            wordTokens = self.wordTokens
        i = 0
        randomOrder = np.random.choice(range(len(wordTokens)), len(wordTokens), replace=False)
        randomOrder = [int(x) for x in randomOrder if x not in excludeList]
        while i < len(wordTokens):
            corpIndex = randomOrder[i]
            result = self._chew_one_stanford(corpIndex)
            yield corpIndex, result
            i += 1
    def _fake_stanTokens_init(self):
        self.stanTokens = [False] * len(self.wordTokens)
    # def _nibble_stanford_ordered(self, )
    def _nibble_stanford(self, chew_generator, n=20):
        for _ in range(n):
            try:
                i, x = next(chew_generator)
            except StopIteration:
                break
            self.stanTokens[i] = x
    def _chew_one_stanford(self, wordToken):
        if isinstance(wordToken, int):
            wordToken = self.wordTokens[wordToken]
        token = " ".join(wordToken)
        if ";;" in token:
            result = False
        else:
            result = output_for_sentence(token)
        return result
    # Fill it in with     