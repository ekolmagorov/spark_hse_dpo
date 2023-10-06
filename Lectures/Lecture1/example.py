import os


if __name__ == '__main__':
    if os.path.exists('/root/data/data.txt'):
        with open('/root/data/data.txt', 'r', encoding='utf-8') as infile:
            result = sum((float(line.strip()) for line in infile))
            print('Sum is equal to : {0:.3f}'.format(result))
    else:
        print('File /root/data/data.txt does not exist')

    print('SECRET_KEY env variable is equal to: {0}'.format(os.environ.get('SECRET_KEY', '')))
