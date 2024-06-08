import codecs

def gtk2utf8(input_file, output_file):
    with codecs.open(input_file, 'r', 'gbk') as f:
        content = f.read()

    with codecs.open(output_file, 'w', 'utf-8') as f:
        f.write(content)
        
if __name__ == '__main__':
    input_file = 'zrj.csv'
    output_file = 'test.csv'
    gtk2utf8(input_file, output_file)
