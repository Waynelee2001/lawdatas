content = open("rag/agent_tools.py", encoding="utf-8").read()

replacements = [
    (
        'f"\u3010get_article \u9519\u8bef\u3011\u672a\u627e\u5230\u5305\u542b"{law_name_keyword}"\u7684\u6cd5\u5f8b"',
        "f'\u3010get_article \u9519\u8bef\u3011\u672a\u627e\u5230\u5305\u542b {law_name_keyword!r} \u7684\u6cd5\u5f8b'",
    ),
    (
        'f"\u3010get_article \u9519\u8bef\u3011\u5728"{law_name_keyword}"\u4e2d\u672a\u627e\u5230\u6761\u6587"{article_num}"\u3002"',
        "f'\u3010get_article \u9519\u8bef\u3011\u5728 {law_name_keyword!r} \u4e2d\u672a\u627e\u5230\u6761\u6587 {article_num!r}'",
    ),
    (
        'f"\u3010get_citing_articles \u9519\u8bef\u3011\u672a\u627e\u5230\u6cd5\u5f8b"{law_name_keyword}""',
        "f'\u3010get_citing_articles \u9519\u8bef\u3011\u672a\u627e\u5230\u6cd5\u5f8b {law_name_keyword!r}'",
    ),
    (
        'f"\u3010get_citing_articles \u9519\u8bef\u3011\u5728"{law_name_keyword}"\u4e2d"',
        "f'\u3010get_citing_articles \u9519\u8bef\u3011\u5728 {law_name_keyword!r} \u4e2d'",
    ),
    (
        'f"\u672a\u627e\u5230\u6761\u6587"{article_num}""',
        "f'\u672a\u627e\u5230\u6761\u6587 {article_num!r}'",
    ),
    (
        'f"\u3010get_cited_articles \u9519\u8bef\u3011\u672a\u627e\u5230\u6cd5\u5f8b"{law_name_keyword}""',
        "f'\u3010get_cited_articles \u9519\u8bef\u3011\u672a\u627e\u5230\u6cd5\u5f8b {law_name_keyword!r}'",
    ),
    (
        'f"\u3010get_cited_articles \u9519\u8bef\u3011\u5728"{law_name_keyword}"\u4e2d"',
        "f'\u3010get_cited_articles \u9519\u8bef\u3011\u5728 {law_name_keyword!r} \u4e2d'",
    ),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print("Fixed:", old[:40])
    else:
        print("NOT FOUND:", old[:40])

# Fix TOOL_SCHEMAS descriptions
content = content.replace(
    '"\u6c11\u6cd5\u5178"\u3001"\u516c\u53f8\u6cd5"',
    "\u6c11\u6cd5\u5178\u3001\u516c\u53f8\u6cd5",
)

import ast
import sys

try:
    ast.parse(content)
    open("rag/agent_tools.py", "w", encoding="utf-8").write(content)
    print("syntax OK - file saved")
except SyntaxError as e:
    lines = content.splitlines()
    print(f"Still broken at line {e.lineno}:")
    print(repr(lines[e.lineno - 1]))
    sys.exit(1)
