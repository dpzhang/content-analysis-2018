import random
import pandas
import nltk
from nltk.corpus import stopwords #For stopwords
import requests
import bs4
import re

def scrapOneChapter(url):
    # get the content of url
    ContentRequest = requests.get(url)
    # bs4 package to parse the html 
    ContentSoup = bs4.BeautifulSoup(ContentRequest.text, 'html.parser') 
    # contentPTags is a collection of all tags with type 'p'
    contentFind = ContentSoup.body.findAll('div', itemprop="articleBody")
    contentPTags = contentFind[0].findAll('p')
    # get rid of top and botton (previous and next page instructions)
    contentPTags = contentPTags[2:-1]
    content_byPtags = [ptag.text for ptag in contentPTags]
    scrapedChapter = " ".join(content_byPtags)
    scrapedChapter = re.sub(r'([c|C]hapter [0-9]+ –) ', '', scrapedChapter)
    return scrapedChapter


def scrapOneChapter2(url):
    # get the content of url
    ContentRequest = requests.get(url)
    # bs4 package to parse the html 
    ContentSoup = bs4.BeautifulSoup(ContentRequest.text, 'html.parser')
    
    # contentPTags is a collection of all tags with type 'p'
    contentFind = ContentSoup.body.findAll('div', itemprop="articleBody")
    contentPtags = ContentSoup.find_all('p', {'dir':'ltr'})
    content_byPtags = [ptag.text for ptag in contentPtags]
    scrapedChapter = ' '.join(content_byPtags)
    scrapedChapter = re.sub(r'([c|C]hapter [0-9]+ –) ', '', scrapedChapter)
    return scrapedChapter
    

def scrapOneBook(bookInfo):
    '''
    bookInfo = [bookName, bookCountry]
    urlInfo = [urlFirst, urlSecond]
    '''
    bookName, bookAuthor, bookCountry, urlFirst, urlSecond= bookInfo
    chapterNum = 51
    urlTemplate = "http://www.wuxiaworld.com/{}-index/{}-chapter-{}/"
    if bookName == 'Talisman Emperor':
        urlTemplate = "http://www.wuxiaworld.com/{}-index/{}{}/"
    if bookName == 'Heavenly Jewel Change':
        urlTemplate = 'http://www.wuxiaworld.com/{}-index/{}-chapter-{}-1/'
    if bookName == 'Coiling Dragon':
        urlTemplate = "http://www.wuxiaworld.com/{}-html/{}-chapter-1/"
    if bookName == 'Stellar Transformations':
        chapterNum = 48
    if bookName == 'The Godsfall Chronicles':
        chapterNum = 46
    if bookName == '7 Killers':
        chapterNum = 8 + 1
    if bookName == 'Child of Light':
        chapterNum = 39 + 1
    if bookName == "Dragon King With Seven Stars":
        chapterNum = 25 + 1
    if bookName == 'Heroes Shed No Tears':
        chapterNum = 18 + 1
    if bookName == 'Horizon, Bright Moon, Sabre':
        chapterNum = 24 + 1
        
    bookDict = {'name':[bookName],
                'author':[bookAuthor],
                'country':[bookCountry],
                'text':[]}

    print("Scraping: {}".format(bookName))
    for i in range(1, chapterNum): ############
        chapterURL = urlTemplate.format(urlFirst, urlSecond, i)
        if bookName == 'Blue Phoenix':
            bookDict['text'].append(scrapOneChapter2(chapterURL))
            print('    Chapter {} done!'.format(i))
        else:
            bookDict['text'].append(scrapOneChapter(chapterURL))
            print('    Chapter {} done!'.format(i))
    bookDict['text'] = ' '.join(bookDict['text'])
    
    return pandas.DataFrame(bookDict)


def scrapAllBook():
    bookInfo = [ 
        ["A Record of a Mortal's Journey to Immortality", 'Wang Yu', 'China', 'rmji', 'rmji'],
        ['Ancient Strengthening Technique', 'I Am Superfluous', 'China', 'ast', 'ast'],
        ["Emperor's Domination", 'Yanbi Xiaosheng', 'China', 'emperor', 'emperor'],
        ['Imperial God Emperor', 'Warring Blades', 'China', 'ige', 'ige'],
        ['Lord of All Realm', 'Ni Cangtian', 'China', 'loar', 'loar'],
        ['Monarch of Evernight', 'Misty Rain of Jiangnan', 'China', 'men', 'men-volume-2'],
        ['Renegade Immortal', 'Er Geng', 'China', 'renegade', 'renegade'],
        ['Spirit Realm', 'Against the Heavens', 'China', 'sr', 'sr'],
        ['Talisman Emperor', 'Xiao Jingyu', 'China', 'talisman-emperor', 'te-ch'],
        ['The Grandmaster Strategist', 'Sui Bo Zhu Liu', 'China', 'tgs', 'tgs'],
        ['A Will Eternal', 'Er Geng', 'China', 'awe', 'awe'],
        ['Battle Through the Heavens', 'Tan Can Tu Dou', 'China', 'btth', 'btth'],
        ['Gate of Revelation', 'Dancing', 'China', 'gor', 'gor'],
        ['Invincible', 'Shen Jian', 'China', 'invincible', 'invincible'],
        ['Martial God Asura', 'Kindhearted Bee', 'China', 'mga', 'mga'],
        ['Perfect World', 'Chen Dong', 'China', 'pw', 'pw'],
        ['Skyfire Avenue', 'Tang Jia San Shao', 'China', 'sfl', 'skyfire-avenue'],
        ['Spirit Vessel', 'Jiu Dang Jia', 'China', 'spiritvessel', 'spirit-vessel'],
        ['The Great Ruler', 'Tian Can Tu Dou', 'China', 'tgr', 'tgr'],
        ['Upgrade Specialist in Another World', 'Endless Sea of Clouds', 'China', 'usaw', 'usaw-book-1'],
        ['Against the Gods', 'Ni Tian Xie Shen', 'China', 'atg', 'atg'],
        ['The Charm of Soul Pets', "Fish Sky", 'China', 'tcosp', 'tcosp'],
        ['Desolate Era', 'I Eat Tomatoes', 'China', 'desolate-era', 'de-book-18-'],
        ['Heavenly Jewel Change', 'Tang Jia San Shao', 'China', 'hjc', 'hjc'],
        ['Legend of the Dragon King', 'Tang Jia San Shao', 'China', 'ldk', 'ldk'],
        ['Martial World', 'Can Jian Li De Niu', 'China', 'martialworld', 'mw'],
        ['Rebirth of the Thief Who Roamed the World', 'Mad Snail', 'China', 'rebirth', 'rebirth'],
        ['Sovereign of the Three Realms', 'Plow Days', 'China', 'sotr', 'sotr'],
        ['Tales of Demons and Gods', 'Mad Snail', 'China', 'tdg', 'tdg'],
        ['TranXending Vision', 'Li Xianyu', 'China', 'tv', 'tranxending-vision'],
        ['7 Killers', 'Gu Long', 'China', 'master', '7-killers'],
        ['Child of Light', 'Tang Jia San Shao', 'China', 'col', 'col-volume-10'],
        ['Wu Dong Qian Kun', 'Tian Can Tu Dou', 'China', 'wdqk', 'wdqk'],
        ['Coiling Dragon', 'I Eat Tomatoes', 'China', 'cdindex', 'book-8'],
        ['Dragon King With Seven Stars', 'Gu Long', 'China', 'master', 'dkwss'],
        ['I Shall Seal the Heavens', 'Er Geng', 'China', 'issth', 'issth-book-1'],
        ['The Godsfall Chronicles', 'Tipsy Wanderer', 'China', 'godsfall', 'godsfall-book-1'],
        ['Stellar Transformations', 'I Eat Tomatoes', 'China', 'st', 'st-book-10'],
        ['Warlock of the Magus World', "Wen Chao Gong", "China", 'wmw', 'wmw'],
        ['Horizon, Bright Moon, Sabre', 'Gu Long', 'China', 'tymyd', ''],
    
        ['Dragon Maken War', 'Kim Jae-Han', 'Korea', 'dragonmakenwar', 'dmw'],
        ['Infinite Competitive Dungeon Society', 'Toika', 'Korea', 'icds', 'icds'],
        ['Overgeared', 'Park Saenal', 'Korea', 'overgeared', 'og'],
        ['The Book Eating Magician', 'Mekenlo', 'Korea', 'bem', 'bem'],
        ['Acquiring Talent in a Dungeon', 'Mibantan', 'Korea', 'atd', 'atd'],
        ['Breakers', 'Chwiryong', 'Korea', 'breakers', 'breakers'],
        ['Emperor of Solo Play', 'D-Dart', 'Korea', 'emperorofsoloplay', 'esp'],
        ['God of Crime', 'Han Yeoul', 'Korea', 'godofcrime', 'goc'],
        ['I am Sorry For Being Born In This World!', 'PALOW', 'Korea', 'isbbtw', 'isbbtw'],
        ['Praise the Occ!', 'Lee Jungmin', 'Korea', 'pto', 'pto'],
        ["Seoul Station's Necromancer", "Crown", "Korea", "ssn", "ssn"],

        ['Blue Phoenix', 'Tinalynge', 'US', 'bp', 'bp'],
        ['Legends of Ogre Gate', 'Deathblade', 'US', 'legends-of-ogre-gate', 'loog'],
        ['The Divine Elements', 'Daman', 'US', 'tde', 'tde']]
    
    firstBook = bookInfo[0]
    bookDF = scrapOneBook(firstBook)
    for book in bookInfo[1:]:
        nextBook = scrapOneBook(book)
        bookDF = bookDF.append(nextBook)
    bookDFReturn = bookDF.reset_index()
    return bookDFReturn


def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None):
    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)
        
    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)
    
    #We will return a list with the stopwords removed
    return list(workingIter)


def processDF(bookDF, stopwordLst = None, stemmer = None, lemmer = None):
    # tokenize
    bookDF['tokenized_text'] = bookDF['text'].apply(lambda x: nltk.word_tokenize(x))
    bookDF['word_counts'] = bookDF['tokenized_text'].apply(lambda x: len(x))

    # normalize
    bookDF['normalized_tokens'] = bookDF['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst, stemmer, lemmer))
    bookDF['normalized_tokens_count'] = bookDF['normalized_tokens'].apply(lambda x: len(x))
    return bookDF


if __name__ == "__main__":
    # scrap the data
    bookDF = scrapAllBook()

    # prepare to process raw text data
        # stop words
    stop_words_nltk = stopwords.words('english')
    stop_words_nltk += ['chapter', 'prologue', 'part']
        # initialize stemmer and lemmer
    snowball = nltk.stem.snowball.SnowballStemmer('english')
    wordnet = nltk.stem.WordNetLemmatizer()
    # process data
    bookDF = processDF(bookDF, stop_words_nltk, snowball, wordnet)
    bookDF.to_csv('bookDF.csv', index = False, sep = ';')
