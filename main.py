from sound import Sound
from consonant import Consonant
from vowel import Vowel

#modifications
'''elision and insertion on any sound
consonants = lenition and fortition
vowels = monophthongization and diphthongization'''

word = 'klaros'
goal = 'klew'

ipaMatrix = []

test_sound = Sound(0, 'b')
test_sound.display_char()
test_sound.display_voicing()

test_cons = Consonant(1, 'a')
test_cons.display_char()
test_cons.display_voicing()

test_vowel = Vowel(2, 'c')
test_vowel.display_char()
test_vowel.display_voicing()