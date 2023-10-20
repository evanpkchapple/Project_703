class Sound(object):
    def __init__(self, voicing, char_rep):
        self.voicing = voicing
        self.char_rep = char_rep

    #display functions for testing while I learn python polymorphism

    def display_char(self):
        print(self.char_rep)

    def display_voicing(self):
        print(self.voicing)

    #sorting functions into the classes to remember
    def elision(self):
        pass

    def insertion(self):
        pass