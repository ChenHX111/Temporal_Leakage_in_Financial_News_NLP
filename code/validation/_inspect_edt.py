"""Inspect EDT JSON structure - parse first complete record."""
import json

with open(r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package\data\external\evaluate_news.json', 'rb') as f:
    chunk = f.read(50000)
s = chunk.decode('utf-8', errors='replace').lstrip()[1:]
depth = 0
in_str = False
escape = False
end = -1
for i, ch in enumerate(s):
    if escape:
        escape = False
        continue
    if ch == '\\':
        escape = True
        continue
    if ch == '"':
        in_str = not in_str
        continue
    if in_str:
        continue
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
print(f'Object ends at byte {end}')
d = json.loads(s[:end])
print('Keys:', list(d.keys()))
for k, v in d.items():
    sv = str(v)
    print(f'  {k} ({type(v).__name__}): {sv[:150]}')
