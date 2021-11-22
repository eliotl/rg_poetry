#!/usr/bin/env python

import AppKit
import re
import sys
# in order to be able to run this, pip install pyobjc in your venv.
# Explaination of the phonemes: https://developer.apple.com/library/content/documentation/UserExperience/Conceptual/SpeechSynthesisProgrammingGuide/Phonemes/Phonemes.html
# from corpusclass import scan_from_parse 

vowls = {
    'AA': 'father',
    'AE': 'bat',
    'AO': 'caught',
    'AW': 'bout',
    'AX': '(a)bout',
    'AY': 'bite',
    'EH': 'bet',
    'EY': 'bait',
    'IH': 'bit',
    'IY': 'beet',
    'IX': 'ros(e)s',
    'UH': 'book',
    'UW': 'boot',
    'UX': 'bud',
    'OW': 'boat',
    'OY': 'boy',
}


vowls2 = {
    'AA': 'father',
    'AE': 'bat',
    'EY': 'bait',
    'EH': 'bet',
    'IY': 'beet',
    'IX': 'roses',
    'IH': 'bit',
    'AY': 'bite',
    'AO': 'caught',
    'OW': 'boat',
    'UH': 'book',
    'AW': 'bout',
    'AX': 'about',
    'UW': 'boot',
    'OY': 'boy',
    'UX': 'bud',
}

consonants2 = {
    'b': 'bin',
    'd': 'din',
    'f': 'fin',
    'J': 'jump',
    'k': 'kin',
    'l': 'limb',
    'm': 'mat',
    'n': 'nap',
    'p': 'pin',
    'r': 'ran',
    's': 'sin',
    't': 'tin',
    'v': 'van',
    'w': 'wet',
    'y': 'yet',
    'z': 'zoo',
    'g': 'gain',
    'N': 'tang',
    'h': 'hat',
    'C': 'chin',
    'D': 'them',
    'S': 'shin',
    'T': 'thin',
    'Z': 'measure',
}

    # JNCDSTZ

# I THINK IT'S PRETTY WRONG that try == trAY, not try == CrAY

consonants = {
    'b': 'bin',
    'C': 'chin',
    'd': 'din',
    'D': 'them',
    'f': 'fin',
    'g': 'gain',
    'h': 'hat',
    'J': 'jump',
    'k': 'kin',
    'l': 'limb',
    'm': 'mat',
    'n': 'nap',
    'N': 'tang',
    'p': 'pin',
    'r': 'ran',
    's': 'sin',
    'S': 'shin',
    't': 'tin',
    'T': 'thin',
    'v': 'van',
    'w': 'wet',
    'y': 'yet',
    'z': 'zoo',
    'Z': 'measure',
}


def get_synth_parse(text):
    synth = AppKit.NSSpeechSynthesizer.alloc().init()
    return synth.phonemesFromText_(text)

def synth_parse(text, synth):
    return synth.phonemesFromText_(text)

def get_synth():
    return AppKit.NSSpeechSynthesizer.alloc().init()

def phones_in_meter(phoneString, meter):
    backwardsMeter = list(reversed(meter))
    backwardsMeter = [x if x != "0" else "" for x in backwardsMeter]
    backwardsPhones = list(filter(None, re.split(r"([12]?[AEIOU][AEOWHYX])", phoneString)))[::-1]
    backwardsOutput = []
    for stress in backwardsMeter:
        for i, phoneme in enumerate(backwardsPhones):
            if phoneme[0] in "12AEIOU":
                backwardsOutput.append(stress + phoneme[-2:])
                break
            else:
                backwardsOutput.append(phoneme)
        backwardsPhones = backwardsPhones[i+1:]
    backwardsOutput += backwardsPhones
    meteredString = "".join(backwardsOutput[::-1])
    return meteredString

def text_in_meter(text, meter, synth = None):
    if synth == None:
        synth = get_synth()
    phones = synth.phonemesFromText_(text)
    meteredPhones = phones_in_meter(phones, meter)
    return meteredPhones


if __name__ == '__main__':

    synth = AppKit.NSSpeechSynthesizer.alloc().init() # This gets you a speech synthesizer
    scanning = False

    from meter_search.corpusclass import scan_from_parse_space
    
    while True:
        inp = input("Phrase to parse: ")
        
        if inp in ["",'.']:
            break

        if inp == '~~':
            scanning = True
            print('Returning Scans')
            continue

        text = inp
        phones = synth.phonemesFromText_(text)

        if scanning:
            phones = scan_from_parse_space(phones)

        print(phones)
        