import re
import os

DATAPATH = 'data'
ORIGINPATH = 'origin_data'
PROCESSEDPATH = 'data'

def read_file_list(folder_path):
    r_l = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            r_l.append(os.path.join(root, fname))
    return r_l


def _extract_body(file_name):
    txt = open(file_name, encoding='cp949').read()
    return '.'.join(re.findall(r"<body>.*</body>", txt, re.DOTALL))

def _extract_text(txt, file_name):
    # for head in re.findall(r"<head>.+</head>", txt, re.DOTALL):
    #     print("-"*20)
    #     print(head)
    #     temp = head.split("\n")[1:-1]
    #     print("-"*20)
    #     print(temp)
    f = open("/".join([DATAPATH, PROCESSEDPATH, file_name]), 'w')

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(txt, 'html.parser')
    _str = ""
    for temp in soup.find_all('head'):
        temp = str(temp)
        
        for t in temp.split("\n")[1:-1]:
            _str += "\t".join(t.split('\t')[1:])
            _str += "\t"
        _str += "\n"
    f.write(_str)

    _str = ""
    for temp in soup.find_all('p'):
        temp = str(temp)
        
        for t in temp.split("\n")[1:-1]:
            _str += "\t".join(t.split('\t')[1:])
            _str += "\t"
        _str += "\n"
    f.write(_str)


def preporcess():
    try:
        os.mkdir("/".join([DATAPATH, PROCESSEDPATH]))
    except:
        pass

    file_list = read_file_list("/".join([DATAPATH, ORIGINPATH]))
    for i in file_list:
        print(i)
        _extract_text(_extract_body(i), i.split('/')[2])

preporcess()
