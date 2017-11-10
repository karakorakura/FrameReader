
import ahocorasick
# dictionary = ['ok','i','am','human','good', 'morning','evening','day','night','sleep','work','how', 'are' , 'you','hello', 'what', 'how' , 'who' ,'is' , 'are' , 'was', 'when', 'you', 'time','1','2','3','4','5','try','trying' ];
# file1 =  open("word3000.txt","r")

tolerance = 5

file1 =  open("wordlist.txt","r")
dictionary=[]
for word in file1 :
    dictionary.append(word.strip('\n\r'))
print (dictionary)
# make trie
A = ahocorasick.Automaton()
for idx, key in enumerate(dictionary):
   A.add_word(key, (idx, key))
# Now convert the trie to an Aho-Corasick automaton to enable Aho-Corasick search:
A.make_automaton()


# print (dictionary)

charStream = "...";
textStream ="....";
letter = ""
sentenceStream ="... "
printedSentences = []
# while(True):
inputStream = input()
tupleResults = []
for inputCharacter in inputStream:
    letterAddedFlag=0
    # inputCharacter = input();
    if inputCharacter != charStream[-1]:
        charStream = inputCharacter

    if len(charStream) >=tolerance:
        letter = inputCharacter
        textStream+=letter
        letterAddedFlag=1
        charStream = inputCharacter

    charStream+=inputCharacter;

    if letterAddedFlag==1:
        # tupleResults = aho_corasick(textStream,dictionary)
        haystack = textStream
        tupleResults=[]
        for end_index, (insert_order, original_value) in A.iter(haystack):
            start_index = end_index - len(original_value) + 1
            tupleResults.append((start_index, end_index, (insert_order,  haystack[start_index:end_index+1])))
            # print((start_index, end_index, (insert_order, original_value)))
            assert haystack[start_index:start_index + len(original_value)] == original_value
        # print (textStream)
        prevIndex=-1
        if len(tupleResults)>=10:
            prevstart_index, prevend_index, (previnsert_order, prevword) = (0,0,(0," "))
            tupleResults.sort()
            # print (tupleResults)
            prevIndex = -1
            for start_index, end_index, (insert_order, word) in tupleResults:
                if start_index==prevstart_index:
                    prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
                    continue;#skip
                elif start_index<=prevend_index:
                    continue;#skip
                sentenceStream+=prevword
                sentenceStream+=" "
                print(prevword),
                prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
            sentenceStream+=prevword
            sentenceStream+=" "
            print(prevword),
            tupleResults=[]
            haystack=haystack[prevend_index+1:]
            textStream = textStream[prevend_index+1:]


## last iteration
prevstart_index, prevend_index, (previnsert_order, prevword) = (0,0,(0," "))
tupleResults.sort()
# print (tupleResults)
prevIndex = -1
for start_index, end_index, (insert_order, word) in tupleResults:
    if start_index==prevstart_index:
        prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
        continue;#skip
    elif start_index<=prevend_index:
        continue;#skip
    sentenceStream+=prevword
    sentenceStream+=" "
    print(prevword),
    prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
sentenceStream+=prevword
sentenceStream+=" "
print(prevword),

printedSentences.append(sentenceStream)
sentenceStream = " "
print(' ');
textStream = textStream[prevIndex+1:]
print (printedSentences)
