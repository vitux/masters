# download source codes via proxy
import datetime
import errno
import json
import os
import random
import sys
import time
import traceback

from threading import Lock
from multiprocessing.dummy import Pool as ThreadPool, current_process

import requests
import user_agent

from lxml import html

API_BASE = 'http://codeforces.com/api/{}'
PRINT_LOCK = Lock()
FAIL_LOCK = Lock()
FAIL_COUNT = 0
FAIL_COUNT_LIMIT = 100

SOLUTIONS_LINK = 'https://codeforces.com/problemset/status/{}/problem/{}/page/{}?order=BY_PROGRAM_LENGTH_ASC'
TITLES_TO_RANK = {
    'Unrated,': 0,
    'Newbie': 1,
    'Pupil': 2,
    'Specialist': 3,
    'Expert': 4,
    'Candidate Master': 5,
    'Master': 6,
    'International master': 7,
    'Grandmaster': 8,
    'International Grandmaster': 9,
    'Legendary grandmaster': 10,
}


USER_AGENTS = [user_agent.generate_user_agent() for _ in range(100)]
PROXY_LOCK = Lock()
PREVIOUS_PROXY_INDEX = -1 % len(PROXIES)


def get_proxy():
    global PREVIOUS_PROXY_INDEX
    with PROXY_LOCK:
        PREVIOUS_PROXY_INDEX += 1
        PREVIOUS_PROXY_INDEX %= len(PROXIES)
        return PROXIES[PREVIOUS_PROXY_INDEX]


def get_data(request, timeout=10, is_api=True, proxy=None, agent=None):

    if is_api:
        request = API_BASE.format(request)

    ITERATION_COUNT = 7
    iteration = 0
    fail_iterations = 0
    status_403_iterations = 0
    proxy = '127.0.0.1:24000'
    while True:
        try:
            if proxy:
                resp = requests.get(request, timeout=timeout, proxies={'http': proxy, 'https': proxy}, headers={'User-Agent': agent})
            else:
                resp = requests.get(request, timeout=timeout, headers={'User-Agent': agent})
            if resp.status_code == 200:
                if is_api:
                    if resp.json()['status'] == 'OK':
                        return resp.json()
                else:
                    return resp
            else:
                if resp.status_code == 403:
                    status_403_iterations += 1
                    sys.stderr.write('403 ({}), total: {}, url: {}\n'.format(status_403_iterations, iteration, request))
                    if iteration != ITERATION_COUNT - 1:
                        time.sleep(min(60 * 30, 60 * 2 ** iteration) / (1 + random.random() * 0.3))
                else:
                    fail_iterations += 1
                with open('errors/{}.html'.format(str(datetime.datetime.now()).replace(' ', 'T')), 'w') as err:
                    err.write('{}\t{}\n{}'.format(resp.status_code, iteration, resp.text.encode('utf-8')))
        except Exception, e:
            fail_iterations += 1
            with PRINT_LOCK:
                if iteration == 6:
                    sys.stderr.write('retrying {}, {}, err: {}\n'.format(request, iteration, e))
        time.sleep(min(2 ** iteration, 60) / (1 + random.random() * 0.3))
        iteration += 1
        if fail_iterations >= ITERATION_COUNT:
            break

    with PRINT_LOCK:
        sys.stderr.write('retry limit reached, url: {}\n'.format(request))
    with FAIL_LOCK:
        global FAIL_COUNT
        FAIL_COUNT += 1
    time.sleep(random.random() * 10)
    return None

# def parse_archive():
#     r = requests.get('https://codeforces.com/problemset')
#     tree = html.fromstring(r.text)
#     problems_nodes = tree.xpath('//table[@class="problems"]/tr')
#     for problem in problems_nodes:
#         problem
#     x = 1


def download_tasks_with_tags():
    # r = requests.get('https://codeforces.com/api/problemset.problems')
    # problems = get_data('problemset.problems')
    # with open('tasks_info.json', 'w') as out:
    #     json.dump(problems, out)
    with open('tasks_info.json') as data_file:
        return json.load(data_file)


def download_source_code(url, current_proxy, agent):
    time.sleep(0.38 + random.random() * 0.02)
    for iteration in range(3):
        r = get_data('https://codeforces.com{}'.format(url), is_api=False, proxy=current_proxy, agent=agent)
        if r is None:
            raise Exception('')
        tree = html.fromstring(r.text)
        try:
            src = tree.xpath('//pre[@id="program-source-text"]/text()')[0]
            return src
        except:
            time.sleep(0.3 *(iteration + 1))
            pass
    return ''


def download_ajax_source_code(contestId, problem_index, submissionid):
    ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.103 YaBrowser/18.7.0.2767 (beta) Yowser/2.5 Safari/537.36'
    BASE_URL = 'https://codeforces.com/contest/{}/status/{}'.format(contestId, problem_index)
    r1 = requests.get(BASE_URL, headers={'User-Agent': ua})
    tree = html.fromstring(r1.text)
    csrf = tree.xpath('//meta[@name="X-Csrf-Token"]')[0].attrib['content']

    cookies = {
        'JSESSIONID': r1.cookies['JSESSIONID'],
        '39ce7': r1.cookies['39ce7'],
        '__utma': '71512449.933383861.1552169197.1552169197.1552169197.1',
        '__utmc': '71512449',
        '__utmz': '71512449.1552169197.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)',
        '__utmt': '1',
        '__utmb': '71512449.2.10.1552169197',
    }

    headers = {
        'Origin': 'https://codeforces.com',
        'Accept-Encoding': 'gzip, deflate, br',
        'X-Csrf-Token': csrf,
        'Accept-Language': 'ru,en;q=0.9,be;q=0.8',
        'User-Agent': ua,
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': BASE_URL,
        'X-Requested-With': 'XMLHttpRequest',
        'Connection': 'keep-alive',
    }

    data = {
        'submissionId': submissionid,
        'csrf_token': csrf,
    }

    response = requests.post('https://codeforces.com/data/submitSource', headers=headers, cookies=cookies, data=data)#, proxies={'https': PROXIES[0], 'http':PROXIES[0]})
    source = response.json()['source']
    return source


PAGE_COUNT = 7


def get_page_indexes(last_page):
    if last_page <= PAGE_COUNT:
        return range(1, last_page + 1)
    middle = last_page / 2
    left = max(middle - PAGE_COUNT / 2, 1)
    right = left + PAGE_COUNT - 1
    return range(left, right + 1)


def save_html_for_debug(r):
    filename = str(datetime.datetime.now()).replace(' ', 'T')
    with open('errors/{}.html'.format(filename), 'w') as out:
        out.write(r.text.encode('utf-8'))
    return filename


def process_one_problem(problem, debug=False):
    global FAIL_COUNT
    if FAIL_COUNT > FAIL_COUNT_LIMIT:
        sys.stderr.write('FAIL_COUNT: {}\n'.format(FAIL_COUNT))
        time.sleep(60 * 20)  # wait 20 minutes
        FAIL_COUNT = 0
        return
    time.sleep(1 + random.random() * 3)
    pid = 5 if debug else int(current_process().name.split('-')[1]) - 1
    current_proxy = '127.0.0.1:24000'
    agent = USER_AGENTS[pid]
    try:
        with PRINT_LOCK:
            sys.stderr.write('processing problem {}{}\n'.format(problem['contestId'], problem['index']))
        r = get_data(SOLUTIONS_LINK.format(problem['contestId'], problem['index'], 1), is_api=False, proxy=current_proxy, agent=agent)
        tree = html.fromstring(r.text)

        if not tree.xpath('//a[@class="view-source"]'):
            raise Exception('no solutions found on page')

        last_page_node = tree.xpath('//li/span[@class="page-index"]/a/text()')
        last_page = int(last_page_node[-1]) if last_page_node else 1

        task_id = '{}/{}'.format(problem['contestId'], problem['index'])
        print 'page_index {}'.format(task_id), last_page, get_page_indexes(last_page)

        for page_index in get_page_indexes(last_page):

            r = get_data(
                SOLUTIONS_LINK.format(problem['contestId'], problem['index'], page_index),
                is_api=False,
                proxy=current_proxy,
                agent=agent,
            )
            tree = html.fromstring(r.text)
            solutions = tree.xpath('//a[@class="view-source"]')

            for solution in solutions:
                if FAIL_COUNT > FAIL_COUNT_LIMIT:
                    sys.stderr.write('FAIL_COUNT: {}\n'.format(FAIL_COUNT))
                    return
                submissionid = solution.attrib.get('submissionid', None)  # solution.xpath('text()')
                try:
                    try:
                        title = solution.xpath('../../td[3]/a')[0].attrib['title'].rsplit(' ', 1)[0]
                    except IndexError, e:
                        if solution.xpath('../../td[3]/span'):
                            continue  # team solution
                        else:
                            raise e
                    if TITLES_TO_RANK[title] < 5:
                        continue
                    lang = solution.xpath('../../td[5]/text()')[0]
                    if 'C++' not in lang:
                        continue

                    # src_code = download_ajax_source_code(problem['contestId'], problem['index'], submissionid)

                    filename = 'source_codes/{}/{}/{}.cpp'.format(
                        problem['contestId'],
                        problem['index'],
                        submissionid,
                    )
                    if task_id not in filename:
                        raise Exception('{} not in {}'.format(task_id, filename))
                    if os.path.exists(filename):
                        continue
                    src_code = download_source_code(solution.attrib['href'], current_proxy, agent)
                    if not os.path.exists(os.path.dirname(filename)):
                        try:
                            os.makedirs(os.path.dirname(filename))
                        except OSError as exc:  # Guard against race condition
                            if exc.errno != errno.EEXIST:
                                raise
                    with open(filename, 'w') as out:
                        out.write(src_code.encode('utf-8'))
                except Exception, e:
                    with PRINT_LOCK:
                        html_filename = save_html_for_debug(r)
                        sys.stderr.write('submissionid {}, err: {}, filename: {}\n{}\n'.format(
                            submissionid,
                            e,
                            html_filename,
                            traceback.format_exc(),
                        ))
    except Exception, e:
        with PRINT_LOCK:
            sys.stderr.write('problem: {}{}, err: {}\n{}\n'.format(
                problem['contestId'],
                problem['index'],
                e,
                traceback.format_exc(),
            ))
    with PRINT_LOCK:
        sys.stderr.write('finish problem {}{}\n'.format(problem['contestId'], problem['index']))


def main():
    start = 0
    end = 5000
    problems = download_tasks_with_tags()
    problems_batch = problems['result']['problems'][start:end]
    thread_pool = ThreadPool(10)
    list(thread_pool.imap_unordered(process_one_problem, problems_batch))


if __name__ == '__main__':
    main()
