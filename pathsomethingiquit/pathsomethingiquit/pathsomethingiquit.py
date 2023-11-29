def the_data_function(path):
    f = open('path', encoding="utf-8")
    source_words = []
    target_words = []

    in_chars = {}
    out_chars = {}
    inv_target = {}

    #iterate through each line
    for line in f:
        #divide the words into target and source
        word = line.strip().split('$')

        #add to respective lists
        source_words.append(word[0])
        target_words.append(word[1])

    source_unique_chars = set()
    target_unique_chars = set()
    #add each word from source list into the set
    #eliminating duplicaate chars
    for word in source_words:
        #convert word into char set and update
        source_unique_chars.update(set(word))

    for word in target_words:
        target_unique_chars.update(set(word))

    source_unique_chars = list(source_unique_chars)
    target_unique_chars = list(target_unique_chars)
    #print(unique_chars)

    #create dictionary of the unique chars in list of source words
    #with key=char and val=index
    for i, char in enumerate(source_unique_chars):
        in_chars[char] = i

    for i, char in enumerate(target_unique_chars):
        out_chars[char] = i
        inv_target[i] = char

    '''print(in_chars)
    print('-----')
    print(out_chars)
    print('-----')
    print(inv_target)'''

    #list of tuples for the target and source words
    dataset = list(zip(source_words,target_words))

    return dataset, in_chars, out_chars, inv_target
    


 