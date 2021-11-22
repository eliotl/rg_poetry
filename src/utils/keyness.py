import numpy as np
import re

from utils.utils import flatten_list

def keyness(file1, file2 = '/Users/eliotlinton/Python/Poetry-Tools/input/gf2.txt'):

	# EMMA = target file
	# SENSE = base file

    from sklearn.feature_extraction.text import CountVectorizer

    # Raw texts is a list of length two... where raw_text[0] is the target file and raw_text[1] is the base rates
    raw_texts = []

    if isinstance(file1, list):
        raw_texts.append(' '.join(flatten_list(file1)))
    else:
        with open(file1) as f:
            raw_texts.append(f.read())
    with open(file2) as f:
        raw_texts.append(f.read())
    raw_texts[0] = re.sub(r'^https?:\/\/.*[\r\n]*', '', raw_texts[0], flags=re.MULTILINE)

    vectorizer = CountVectorizer(input='content')
    dtm = vectorizer.fit_transform(raw_texts)
    vocab = np.array(vectorizer.get_feature_names())

    dtm = dtm.toarray()
    rates = 1000 * dtm / np.sum(dtm, axis=1, keepdims=True)

    emma_indices, sense_indices = [], []

    emma_indices.append(0)
    sense_indices.append(1)

    emma_rates = rates[emma_indices, :]
    sense_rates = rates[sense_indices, :]

    emma_rates_avg = np.mean(emma_rates, axis=0)
    sense_rates_avg = np.mean(sense_rates, axis=0)

    distinctive_indices = (sense_rates_avg == 0) & (emma_rates_avg != 0)

    ranking2 = np.argsort(emma_rates_avg[distinctive_indices])[::-1]

    # This is a little slower than it needs to be
    # I would prefer the scale to be less skewed: "hedgehog" and "croquet" are pretty good... don't need to be so much worse than "hatter"
    scale_rates = 1 + (emma_rates_avg * (0.9/emma_rates_avg[distinctive_indices].max()))
    # scale_rates = emma_rates_avg
    unique = {a:b for a,b in zip(list(vocab[distinctive_indices][ranking2]), list(scale_rates[distinctive_indices][ranking2]))}

    dtm = dtm[:, np.invert(distinctive_indices)]
    rates = rates[:, np.invert(distinctive_indices)]
    vocab = vocab[np.invert(distinctive_indices)]

	# recalculate variables that depend on rates
    emma_rates = rates[emma_indices, :]
    sense_rates = rates[sense_indices, :]
    emma_rates_avg = np.mean(emma_rates, axis=0)
    sense_rates_avg = np.mean(sense_rates, axis=0)

    rates_avg = np.mean(rates, axis=0)
    keyness = (emma_rates_avg - sense_rates_avg) / rates_avg
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order.

    keynessdict = {a:b for a,b in zip(list(vocab), list(keyness)) if b > 0}

    return [keynessdict, unique]