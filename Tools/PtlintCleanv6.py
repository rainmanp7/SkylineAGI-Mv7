import os
import pylint.lint
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
import ast
import astor
import sys
import io
import autopep8
import asyncio
import aiofiles

def configure_logging():
    """Configure logging settings."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_file(fullpath):
    """Check a single Python file for pylint issues."""
    try:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        results = pylint.lint.Run([fullpath, '--disable=C0301,C0114,C0116,C0103'], do_exit=False)
        sys.stdout, sys.stderr = old_stdout, old_stderr
        score = results.linter.stats.global_note
        if score < 10:
            logging.info(f"Found issues in {fullpath}, Score: {score}")
            return fullpath, score
        else:
            logging.info(f"No issues found in {fullpath}")
            return None
    except Exception as e:
        logging.error(f"Error checking file {fullpath}: {e}")
        return None

def prompt_fix_or_skip(file_path):
    """Prompt the user to choose between fixing or skipping a file."""
    choice = input(f"Issue found in {file_path}. Do you want to (f) fix or (s) skip? (f/s): ").strip().lower()
    if choice == 'f':
        return True
    elif choice == 's':
        return False
    else:
        print("Invalid option selected. Skipping...")
        return False

async def fix_imports_and_unused_vars(file_path):
    """Fix imports and remove unused variables in a single file using autopep8 and ast."""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file_handle:
            code = await file_handle.read()
        fixed_code = autopep8.fix_code(code, options={'aggressive': 2})
        tree = ast.parse(fixed_code)
        tree = astor.to_source(RemoveUnusedVariables().visit(tree))
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as file_handle:
            await file_handle.write(tree)
        logging.info(f"Fixed {file_path}")
        return f"Fixed {file_path}"
    except Exception as e:
        logging.error(f"Error fixing file {file_path}: {e}")
        return f"Failed fix {file_path}"

class RemoveUnusedVariables(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        used_vars = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                used_vars.add(child.id)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                used_vars.add(child.id)
        node.body = [n for n in node.body if not (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id not in used_vars for t in n.targets))]
        return node

def collect_results(results):
    """Collect results from the pool map operation."""
    return [result for result in results if result]

def process_files(path, action):
    """Process files based on the specified action."""
    files_to_check = []
    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith('.py'):
                fullpath = os.path.join(dirpath, file)
                files_to_check.append(fullpath)
    num_workers = min(cpu_count(), len(files_to_check))
    with Pool(processes=num_workers) as pool:
        issues = collect_results(list(tqdm(pool.imap_unordered(check_file, files_to_check), total=len(files_to_check), desc="Checking files")))
    if action == '1':
        return issues
    elif action == '2':
        if issues:
            asyncio.run(fix_files_async(issues))
        return issues
    elif action == '3':
        for file_path, score in issues:
            if prompt_fix_or_skip(file_path):
                asyncio.run(fix_imports_and_unused_vars(file_path))
        return issues
    else:
        print("Invalid option selected.")
        return None

async def fix_files_async(issues):
    """Automatically fix pylint issues using autopep8 and ast asynchronously."""
    tasks = [fix_imports_and_unused_vars(file_path) for file_path, _ in issues]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

def main():
    """Main function to execute the script."""
    configure_logging()
    path = '.'
    action = input("Do you want to list issues (1), fix files (2), or check and fix files interactively (3)? (1/2/3): ").strip()
    issues = process_files(path, action)
    if issues:
        print(f"Total files with issues: {len(issues)}")
        for file_path, score in issues:
            print(f"File: {file_path}, Score: {score}")
    else:
        print("No issues found.")

if __name__ == "__main__":
    main()