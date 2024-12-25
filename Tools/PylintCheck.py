import os
import pylint.lint
from tqdm import tqdm
import autopep8
from multiprocessing import Pool
import logging
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


def check_file(fullpath):
    """Check a single Python file for pylint issues."""
    try:
        results = pylint.lint.Run([fullpath], do_exit=False)
        return (fullpath, results.linter.stats.global_note
            ) if results.linter.stats.global_note < 10 else None
    except Exception as e:
        logging.error(f'Error checking file {fullpath}: {e}')
        return None


def check_files(path):
    """Check all Python files in the given directory and its subdirectories for pylint issues."""
    files_to_check = []
    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith('.py'):
                fullpath = os.path.join(dirpath, file)
                files_to_check.append(fullpath)
    with Pool() as pool:
        issues = list(tqdm(pool.imap(check_file, files_to_check), total=len
            (files_to_check), desc='Checking files'))
    return [issue for issue in issues if issue]


def fix_file(file_path):
    """Automatically fix pylint issues in a single file using autopep8."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            code = file_handle.read()
        fixed_code = autopep8.fix_code(code)
        with open(file_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write(fixed_code)
        return f'Fixed {file_path}'
    except Exception as e:
        logging.error(f'Error fixing file {file_path}: {e}')
        return f'Failed to fix {file_path}'


def fix_files(issues):
    """Automatically fix pylint issues using autopep8."""
    with Pool() as pool:
        results = list(tqdm(pool.imap(fix_file, [file_path for file_path, _ in
            issues]), total=len(issues), desc='Fixing files'))
    for result in results:
        print(result)


def main():
    """Main function to execute the script."""
    path = '.'
    action = input('Do you want to (1) list issues or (2) fix files (1/2): '
        ).strip()
    if action == '1':
        issues = check_files(path)
        if issues:
            print(f'Total files with issues: {len(issues)}')
            for file_path, score in issues:
                print(f'File: {file_path}, Score: {score}')
        else:
            print('No issues found.')
    elif action == '2':
        issues = check_files(path)
        if issues:
            print(f'Total files with issues: {len(issues)}')
            fix_files(issues)
        else:
            print('No issues found.')
    else:
        print('Invalid option selected.')


if __name__ == '__main__':
    main()
