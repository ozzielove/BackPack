#!/usr/bin/env python3
import argparse
import pathlib
import re

def patch_file(path, mutator):
    original = path.read_text(encoding='utf-8')
    updated = mutator(original)
    return original != updated, updated

def fix_sprint_dataclass(src):
    if 'class Sprint' not in src:
        return src
    # Add frozen=True to dataclass
    src = re.sub(r'@dataclass\(([^)]*)\)', 
                 lambda m: f'@dataclass({m.group(1)}, frozen=True)' if 'frozen=True' not in m.group(1) else m.group(0), 
                 src)
    # Change list to tuple
    src = re.sub(r'backlog_items\s*:\s*list', 'backlog_items: tuple', src)
    # Convert to tuple in constructor
    src = re.sub(r'backlog_items=backlog_items', 'backlog_items=tuple(backlog_items)', src)
    return src

def improve_assert_reporting(src):
    if 'test_end_to_end_workflow' not in src or 'traceback.print_exc' in src:
        return src
    # Add traceback import and try/except
    pattern = r'(def\s+test_end_to_end_workflow\([^)]*\):\s*)(.*?)(?=\ndef|\Z)'
    def repl(m):
        body_lines = m.group(2).strip().split('\n')
        indented_body = '\n'.join('        ' + line for line in body_lines if line.strip())
        return f'{m.group(1)}    import traceback\n    try:\n{indented_body}\n    except AssertionError:\n        traceback.print_exc()\n        raise\n'
    return re.sub(pattern, repl, src, flags=re.DOTALL)

def fix_backlog_getter(src):
    if 'return list(' in src:
        return src
    return re.sub(r'return\s+([a-zA-Z_][\w.]*)', r'return list(\1)', src)

PATCH_TABLE = [
    (pathlib.Path('enhanced_sprint_planner.py'), fix_sprint_dataclass),
    (pathlib.Path('integrated_test_system.py'), improve_assert_reporting),
    (pathlib.Path('enhanced_product_backlog.py'), fix_backlog_getter),
]

def main():
    parser = argparse.ArgumentParser(description='Apply Zeus patches')
    parser.add_argument('--dry-run', action='store_true', help='Show planned changes')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        parser.error('Choose --dry-run or --apply')
    
    for path, fixer in PATCH_TABLE:
        if not path.exists():
            print(f'⚠️  {path} not found; skipping')
            continue
        
        changed, new_text = patch_file(path, fixer)
        
        if changed:
            if args.dry_run:
                print(f'Would patch {path.name}')
            else:
                path.write_text(new_text, encoding='utf-8')
                print(f'✅ Patched {path.name}')
        else:
            print(f'ℹ️  No changes needed in {path.name}')
    
    if args.apply:
        print('\n🎉 All patches applied. Run your tests!')

if __name__ == '__main__':
    main()
