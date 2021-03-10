import pickle as pkl
import wikipedia


with open('pkl/freq-kw.pkl', 'rb') as file:
    kws = pkl.load(file)
    
# queue = ['Barack Obama', 'United States', 'World Health Organization', 
#          'Olympic Games', 'financial crisis', 'Donald Trump',
#          'European Union', 'Stock market', 'Super Bowl', 'National Basketball Association']

queue = kws

word2summary = {}
word2links = {}
word_freq = {}
ptr = 0
while ptr < len(queue):
    res = wikipedia.search(queue[ptr])
    ptr += 1
    if len(res) > 0:
        keyword = res[0]
        if keyword in word2summary:
            continue
        try:
            summary = wikipedia.summary(keyword, auto_suggest=False)
            page = wikipedia.page(keyword, auto_suggest=False)
            links = page.links
            word2summary[keyword] = summary
            word2links[keyword] = links
            word_freq[keyword] = word_freq.get(keyword, 0) + 1    
            for word in links:
                word_freq[word] = word_freq.get(word, 0) + 1
                if word in summary:
                    queue.append(word)
            if len(word2summary) % 30 == 0:
                print(len(word2summary))

            if (ptr % 1000) == 0:
                print('begin to save data')
                saved_data = [word2summary, word2links, word_freq]
                with open('pkl/words_info_5k.pkl', 'wb') as file:
                    pkl.dump(saved_data, file)

            if len(word2summary) >= 5000:
                break
        except:
            print('Exception when processing "{}", skipped'.format(keyword))
    

print('begin to save data')
saved_data = [word2summary, word2links, word_freq]
with open('pkl/words_info_5k.pkl', 'wb') as file:
    pkl.dump(saved_data, file)