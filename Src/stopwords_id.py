from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def get_stopwords():
    factory = StopWordRemoverFactory()
    indonesian_stopwords = set(factory.get_stop_words())

    custom_stopwords = {
        'mbg', 'makan', 'bergizi', 'gratis', 'program',
        'yang', 'di', 'dan', 'ini', 'itu',
        "mbak","pak","kak","nya","nih","loh","deh","dong",
        "wkwk","wkwkwk","haha","hahaha",
        "anjing","anjir","bjir","jir",
        "lol","lmao",
        "jadi","buat","sama","apa","bukan","banyak",
        "mau","kalau","kalo","terus","banget",
        "semua","orang"
    }

    return indonesian_stopwords.union(custom_stopwords)