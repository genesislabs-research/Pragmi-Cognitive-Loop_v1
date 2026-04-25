import subprocess
import glob

tests = sorted(glob.glob('/home/claude/work/test_*.py'))
total = 0
for t in tests:
    r = subprocess.run(['python3', t], capture_output=True, text=True, cwd='/home/claude/work')
    last = r.stdout.strip().split('\n')[-1] if r.stdout else 'NO OUTPUT'
    print(f'{t.split("/")[-1]}: {last}')
    if 'passed' in last:
        try:
            n = int(last.split('All ')[1].split(' ')[0])
            total += n
        except Exception:
            pass
print(f'TOTAL: {total} tests')
