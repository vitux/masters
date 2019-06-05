# script to visualise tags for solutions
import json
import requests
import sys
import time

from lxml import html


def get_url(url):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            return r
        except:
            pass
        time.sleep(2 * _)
    else:
        sys.stderr.write('Can\'t load page, try later')


def process(url, tag_info):
    r = get_url(url)
    if r is None:
        return
    tree = html.fromstring(r.text)
    _, _, _, _, _, contest_id, _, task_id, *_ = url.split('/')
    tds = tree.xpath('//tr/td[4]')
    idsa = tree.xpath('//td[@class="id-cell"]/a')
    thitem = tree.xpath('//th[4]')[0]
    thitem.text = 'detected tags'
    for a in tree.xpath('//tr/td[1]/a'):
        a.attrib['href'] = 'https://codeforces.com' + a.attrib['href']
    for ida, td in zip(idsa, tds):
        path = 'source_codes/{}/{}/{}.cpp'.format(contest_id, task_id, ida.text)
        detected_tags = ''
        if path in tag_info:
            info = [tag[0] for tag in tag_info[path] if tag[1] > 20][:5]
            detected_tags = ', '.join(info)
        td.text = str(detected_tags)

    tdsa = tree.xpath('//tr/td[4]/a')
    for td in tdsa:
        td.text = ''
    str_html = html.tostring(tree)
    with open('file.html', 'wb') as out:
        out.write(str_html)



def main():
    with open('tags_by_solution_info.json') as json_data_file:
        tag_info = json.load(json_data_file)
    while True:
        print('enter url:')
        url = input().strip()
        # url = 'https://codeforces.com/problemset/status/39/problem/H'
        # url = 'https://codeforces.com/problemset/status/497/problem/D'
        process(url, tag_info)
        print('http://localhost:63342/mag/tag_detection/file.html')


if __name__ == '__main__':
    main()
