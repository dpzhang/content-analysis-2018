from nltk.corpus import stopwords #For stopwords
import random
import pandas
import nltk
import requests
import bs4
import re

def wordCounter(wordLst):
    wordCounts = {}
    for word in wordLst:
        #We usually need to normalize the case
        wLower = word.lower()
        if wLower in wordCounts:
            wordCounts[wLower] += 1
        else:
            wordCounts[wLower] = 1
    #convert to DataFrame
    countsForFrame = {'word' : [], 'count' : []}
    for w, c in wordCounts.items():
        countsForFrame['word'].append(w)
        countsForFrame['count'].append(c)
    return pandas.DataFrame(countsForFrame)

def accessContent(url):
    # get the content of url
    ContentRequest = requests.get(url)
    # bs4 package to parse the html 
    ContentSoup = bs4.BeautifulSoup(ContentRequest.text, 'html.parser')
    
    # contentPTags is a collection of all tags with type 'p'
    contentPTags = ContentSoup.body.findAll('div', itemprop="articleBody")
    contentPTags = contentPTags[0].findAll('p')
    return contentPTags[3:-2]

bookInfo= {"pw":{"Perfect World":"China"},
           "sotr":{"Sovereign of the Three Realms":"China"},
           "emperor":{"Emperor's Domination":"China"},
           "tgr":{"The Great Ruler":"China"}, 
           "wdqk":{"Wu Dong Qian Kun":"China"}, 
           "loar":{"Lord of All Realms":"China"},
           "bem":{"The Book Eating Magician":"Korea"}, 
           "atg":{"Against the Gods":"China"},
           "tdg":{"Tales of Demons & Gods":"China"}, 
           "ast":{"Ancient Strengthening Technique":"China"}
          }

def scrapOneChapter(url):
    contentSpanTags = accessContent(url)
    # compile sentences
    contentSentences = []
    findSentences = r'([A-Z0-9][^\.\?\!]*[\.\!\?]+)'
    for Tag in contentSpanTags:
        contentSentences.append(re.findall(findSentences, Tag.text))
    contentSentences = [sentence for nestedList in contentSentences for sentence in nestedList]
    return " ".join(contentSentences)

def scrapOneBook(book):
    urlTemplate = "http://www.wuxiaworld.com/{}-index/{}-chapter-{}/"
    #random.seed(1)
    #randomChapters = random.sample(range(1, 200), 10)
    bookDict = {'name':[],
                'country':[],
                'shortName':[],
                'url':[],
                'chapter':[],
                'text':[]}

    print("Scraping")
    for i in range(1, 201):
        print("{}: Chapter{}".format(list(bookInfo[book].keys())[0], i))
        chapterURL = urlTemplate.format(book, book, i)
        bookDict['url'].append(chapterURL)
        bookDict['text'].append(scrapOneChapter(chapterURL))
        bookDict['name'].append(list(bookInfo[book].keys())[0])
        bookDict['shortName'].append(book)
        bookDict['country'].append(list(bookInfo[book].values())[0])
        bookDict['chapter'].append(re.search(r'chapter-[0-9]+', chapterURL).group(0))
        
    return pandas.DataFrame(bookDict)


allBooks = list(bookInfo.keys())
# scrap the first book
tenNovels = scrapOneBook(allBooks[0])
# scrap the rest 
for book in allBooks[1:]:
    otherBook = scrapOneBook(book)
    tenNovels = tenNovels.append(otherBook, ignore_index=True)
tenNovels.to_csv("tenNovels200.csv", index = False)
