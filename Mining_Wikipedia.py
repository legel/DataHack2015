__author__ = 'Jeff'

import urllib2
from bs4 import BeautifulSoup


pages = []
f = open("/Users/Jeff/Desktop/DataHack/list.txt", 'r')
for i in range(70):
    topic = f.readline()
    topic = topic.replace(" ","_")
    print topic

    url = "http://en.wikipedia.org/wiki/" + topic

    response = urllib2.urlopen(url)
    html_doc = response.read()

    soup = BeautifulSoup(html_doc)
    words = soup.get_text().lower().split()

    words_to_ignore = ["that","what","with","this","would","from","your","which","while","these","retrieved"]
    things_to_strip = [".",",","?",")","(","\"",":",";","'s"]
    words_min_size = 4

    page = []
    for word in words:
        for thing in things_to_strip:
            if thing in word:
                word = word.replace(thing,"")
        if word not in words_to_ignore and len(word) >= words_min_size:
            page.append(word)

    pages.append(page)

f.close()

print pages

text_file = open("/Users/Jeff/Desktop/DataHack/page.txt", "w")
for p in pages:
    for w in p:
        text_file.write(str(w.encode("UTF-8")) + " ")
    text_file.write(";"+"\n")
text_file.close()


#END
