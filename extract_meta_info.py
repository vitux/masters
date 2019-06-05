# extract data from locally downloaded files
import datetime
import json
import re
import time
import os
import subprocess


FUNCTION_NAME_REGEXP = re.compile('(inline *)?[a-zA-Z][a-zA-Z0-9_]*( *<[ a-zA-Z0-9\,]*>)? +([a-zA-Z_][a-zA-Z0-9_]*) *\([a-zA-Z0-9,<> +-=]*\) *\{?$')
VARIABLE_REGEXP = re.compile('([a-zA-Z][a-zA-Z0-9_]* *(<[ a-zA-Z0-9\,]*>)? +)(([\*&]?)((\[[ 0-9a-zA-Z_+-\.]*\])?) *[a-zA-Z_][a-zA-Z0-9_]* *((\[[ 0-9a-zA-Z_+-\.]*\])?)(\([^\);]*\)|\{[^\]};]*\})?)( *\, *([\*&]?)((\[[ 0-9a-zA-Z_+-\.]*\])?) *[a-zA-Z_][a-zA-Z0-9_]* *(\[[ 0-9a-zA-Z_+-\.]*\])?(\([^\);]*\)|\{[^\]};]*\})?)* *; *$')
ONE_VARIABLE_REGEXP = re.compile('(([\*&]?)((\[[ 0-9a-zA-Z_+-\.]*\])?) *([a-zA-Z_][a-zA-Z0-9_]*) *((\[[ 0-9a-zA-Z_+-\.]*\])?)(\([^\);]*\)|\{[^\]};]*\})?)')


def get_submissions():
    folders = os.listdir('source_codes')
    for foldername in folders:
        tasks = os.listdir('source_codes/{}'.format(foldername))
        for task in tasks:
            submisssons = os.listdir('source_codes/{}/{}'.format(foldername, task))
            for submissson in submisssons:
                yield 'source_codes/{}/{}/{}'.format(foldername, task, submissson)


class MyException(Exception):
    pass


def get_src_lines(submission_path, remove_comments=True):
    if remove_comments:
        out = subprocess.Popen(
            ['g++', '-fpreprocessed', '-dD', '-E', submission_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = out.communicate()
        if not stdout:
            raise MyException('no stdout for path: {}'.format(submission_path))
        return stdout.split('\n')
    else:
        with open(submission_path) as fl:
            return fl.readlines()


def extractor_binary_search(lines):
    templates = [
        re.compile('while *\( *r *- *l *> *1 *\)'),
        re.compile('(mid|m) *= *\( *l *\+ *r *\) */2'),
        re.compile('while *\( *(lo|l) *<=? *(hi|r) *\)'),
    ]
    for line in lines:
        for template in templates:
            if re.search(template, line):
                return True
    return False


def extractor_bitmasks(lines):
    templates = [
        'bitset<[^.]*> ',
        '__builtin_popcount',
        'for *\((int|size_t) *mask *=',
        '\|=',
        '&=',
        '\(1 <<[^\)]*\) *&',
        '& *\(1 <<[^\)]*\)',
    ]
    for line in lines:
        for template in templates:
            if re.search(template, line):
                return True
    return False


def extractor_string_suffix_structures_partial(lines):
    templates = [
        '(struct|class) *trie',
        '(struct|class) *suffix',
        '(struct|class) *trie',
    ]
    for line in lines:
        for template in templates:
            if re.search(template, line.lower()):
                return True
    return False


def extractor_hashing_partial(lines):
    templates = [
        '(struct|class) *hash',
    ]
    for line in lines:
        for template in templates:
            if re.search(template, line.lower()):
                return True
    return False




def extract_submission_info(submission_path):
    if os.path.getsize(submission_path) == 0:
        return
    _, contest, task, code_id = submission_path.split('/')
    target_path = 'parsed_codes/{}/{}/{}'.format(contest, task, code_id)

    with open(target_path) as fl:
        lines = fl.readlines()

    function_names = []
    variables = []
    function_names_debug = []
    variables_debug = []
    tag_specific_extractors = {
        'binary search': extractor_binary_search,
        'bitmasks': extractor_bitmasks,
        'string suffix structures': extractor_string_suffix_structures_partial,
        'hashing': extractor_hashing_partial,
    }
    result = []
    for key, func in tag_specific_extractors.items():
        if func(lines):
            result.append(key)

    for line in lines:
        line = line.strip()
        match = FUNCTION_NAME_REGEXP.match(line)
        if match:
            function_name = match.group(3)
            if function_name not in ('main', 'if') and line.count('<') == line.count('>'):
                function_names_debug.append({'name': function_name, 'line': line})
                function_names.append(function_name)
        match = VARIABLE_REGEXP.match(line)
        if match:
            blocked_words = ['return', 'else']
            skip = False
            for word in blocked_words:
                if word in line.split():
                    skip = True
                    break
            if skip:
                continue

            matched = match.group(0)[len(match.group(1)):]
            variable_names = [x.group(5) for x in ONE_VARIABLE_REGEXP.finditer(matched)]
            variables_debug.append({'names': variable_names, 'line': line})
            variables.extend(variable_names)

    return {'variables': variables, 'function_names': function_names}


def main():
    res = {}
    ok = 0
    failed = 0
    for ind, submission_path in enumerate(get_submissions()):
        if ind % 10000 == 0:
            print(ind, ok, failed, float(ok) / max(1, ok + failed))
        try:
            submission_info = extract_submission_info(submission_path)
            if submission_info:
                res[submission_path] = submission_info
            ok += 1
        except MyException:
            failed += 1

    # with open('additional_tag_processing.txt', 'w') as out:
    #     out.write(json.dumps(res))
    with open(out_path, 'w') as out:
        out.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    main()
