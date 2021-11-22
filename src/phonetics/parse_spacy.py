import spacy
from spacy.language import Language 
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from phonetics.apple_phonemes import get_synth, synth_parse


@Language.component("newline_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text in ("\n\n", "\n"):
            doc[token.i + 1].is_sent_start = True
    return doc



nlp = spacy.load("en_core_web_sm")

nlp = English()
nlp.add_pipe('sentencizer')
# nlp.add_pipe('sentencizer', config={"punct_chars": ["\n"]})

# nnl = English()
nnl = spacy.load("en_core_web_sm")

nnl.add_pipe('newline_boundaries', before='parser')

tokenizer = Tokenizer(nlp.vocab)

text = ''

doc = nlp(text)
doc.sents

def simple_parser():
    nlp = English()
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('newline_boundaries', before='parser')
    return nlp

def core_parser():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('newline_boundaries', before='parser')
    return nlp

def spacy_parse(input_text, parser = 'core'):
    if parser == 'core':
        nlp = core_parser()
    if parser == 'simple':
        nlp = simple_parser()
    return nlp(input_text)


synth = get_synth()


for sent in doc.sents:
    parse = synth_parse(sent, synth)
    pass


while True:
    from spacy.language import Language 
    from spacy.tokenizer import Tokenizer
    from spacy.lang.en import English

    from src.utils.file_handling import open_file
    from src.phonetics.apple_phonemes import get_synth, synth_parse
    x = open_file('raw/dril')

    grafs = [nnl(graf) for graf in x.split('\n') if graf]
    sents_by_graf = [[str(y) for y in x.sents] for x in grafs]
    sents = [y for x in grafs for y in x.sents]

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('newline_boundaries', before='parser')

    for sentence in doc.sents:
        parse = synth_parse(sentence.text, synth)
        # Needs to be ...

        pass