# calculate statistics on contest
import json
import sys
import time

import requests

from collections import Counter

from lxml import html

API_BASE = 'http://codeforces.com/api/{}'
USERS_ON_PAGE = 100


def get_data(req, timeout=10):
    request = API_BASE.format(req)
    for _ in xrange(7):
        try:
            resp = requests.get(request, timeout=timeout)
            if resp.status_code == 200 and resp.json()['status'] == 'OK':
                return resp.json()
            elif resp.status_code == 400:
                return None
        except Exception, e:
            pass
        print 'retrying'
        time.sleep(2 ** _)
    sys.exit(1)


def get_contests():
    data = get_data('contest.list')
    return filter(lambda x: x['phase'] == 'FINISHED', data['result'])


def get_standings(contest_id):
    print contest_id
    standings = get_data('contest.standings?contestId={}'.format(contest_id))
    return standings
    # return map(lambda x: x['party']['members'][0]['handle'], standings['result']['rows'])


def get_rating_changes(contest_id):
    rating_changes = get_data('contest.ratingChanges?contestId={}'.format(contest_id))
    return rating_changes
    # return map(lambda x: x['party']['members'][0]['handle'], rating_changes['result']['rows'])


def get_user_info():
    all_users_data = get_data('user.ratedList', 500)['result']
    return all_users_data


def get_contest_authors():
    authors = {}
    for ind in xrange(1, 11):
        r = requests.get('http://codeforces.com/contests/page/{}'.format(ind))
        r.raise_for_status()
        tree = html.fromstring(r.text)
        for contest_row in tree.xpath('//table')[1].xpath('tr')[1:]:
            contest_id = contest_row.attrib['data-contestid']
            users = [''.join(item.xpath('.//text()')) for item in contest_row.xpath('td')[1].xpath('a')]
            users_old = contest_row.xpath('td')[1].xpath('a/text()')
            authors[contest_id] = users
    return authors


def main():
    authors = get_contest_authors()
    with open('authors.json', 'w') as authors_file:
        authors_file.write(json.dumps(authors))
    contests = get_contests()
    standings = {}
    print len(contests)
    for ind, contest in enumerate(contests):
        standings[contest['id']] = {
                'standings': get_standings(contest['id']),
                'rating_changes': get_rating_changes(contest['id']),
            }
        if ind % 100 == 0 and ind:
            print ind
            with open('contests_{}.json'.format(ind), 'w') as contests_file:
                contests_file.write(json.dumps(standings, ensure_ascii=False).encode('utf-8'))
            standings = {}

    with open('contests_1000.json', 'w') as contests_file:
        contests_file.write(json.dumps(standings, ensure_ascii=False).encode('utf-8'))


    with open('contests.json', 'w') as contests_file:
        contests_file.write(json.dumps(standings, ensure_ascii=False).encode('utf-8'))

    standings = json.load(open('standings.json'))
    user_info = get_user_info()
    with open('user_info.json', 'w') as out:
        out.write(json.dumps(user_info, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    main()
