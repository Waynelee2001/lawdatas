#!/usr/bin/env python3
"""
法考法条库 —— 激活码生成工具
=====================================
用法：
  python3 keygen_tool.py              # 默认生成 10 个激活码
  python3 keygen_tool.py -n 50        # 生成 50 个激活码
  python3 keygen_tool.py -n 20 -o codes.txt   # 生成并保存到文件
  python3 keygen_tool.py --verify ABCDE-FGHIJ # 验证一个激活码

算法说明：
  - 格式：XXXXX-XXXXX（展示用，实际10个有效字符）
  - 字符集：32个无歧义字符（去掉 0/O/1/I 防混淆）：
    ABCDEFGHJKLMNPQRSTUVWXYZ23456789
  - 结构：前7位随机载荷 + 后3位 HMAC-SHA256 校验
  - 安全性：无密钥则无法伪造（校验成功率 1/32768）
"""

import hmac
import hashlib
import secrets
import argparse
import datetime
import sys

# ============================================================
# ⚠️  密钥必须与 auth_server.py 中的 SECRET_KEY 完全一致！
# ============================================================
SECRET_KEY = 'LawFaKao@2024_HMAC_K3y_#Z9!mX'

# 32个无歧义字符（与后端一致）
CHARSET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'


def _to_base32_str(n: int, length: int) -> str:
    result = []
    for _ in range(length):
        result.append(CHARSET[n % 32])
        n //= 32
    return ''.join(reversed(result))


def generate_key() -> str:
    """生成一个10位激活码（格式：XXXXX-XXXXX）"""
    payload = ''.join(secrets.choice(CHARSET) for _ in range(7))
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    n = int.from_bytes(sig[:3], 'big') % (32 ** 3)
    checksum = _to_base32_str(n, 3)
    raw = payload + checksum
    return f'{raw[:5]}-{raw[5:]}'


def validate_key(key_input: str) -> tuple:
    """
    验证激活码是否合法。
    返回 (True/False, 说明信息)
    """
    key = key_input.strip().upper().replace('-', '').replace(' ', '')

    if len(key) != 10:
        return False, f'长度错误（应为10位，当前 {len(key)} 位）'

    for c in key:
        if c not in CHARSET:
            return False, f'含有非法字符：{c}'

    payload = key[:7]
    checksum = key[7:]

    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    n = int.from_bytes(sig[:3], 'big') % (32 ** 3)
    expected = _to_base32_str(n, 3)

    if checksum != expected:
        return False, f'签名不匹配（期望：{expected}，实际：{checksum}）'

    return True, '✅ 合法有效的激活码'


def main():
    parser = argparse.ArgumentParser(
        description='法考法条库激活码生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-n', '--count', type=int, default=10,
                        help='生成激活码的数量（默认10个）')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='保存到文件路径（可选）')
    parser.add_argument('--verify', type=str, default=None,
                        help='验证一个激活码是否合法')
    args = parser.parse_args()

    # —— 验证模式 ——
    if args.verify:
        ok, msg = validate_key(args.verify)
        status = '✅ 合法' if ok else '❌ 无效'
        print(f'\n激活码：{args.verify.upper()}')
        print(f'结  果：{status}')
        print(f'说  明：{msg}\n')
        sys.exit(0 if ok else 1)

    # —— 生成模式 ——
    count = max(1, min(args.count, 10000))  # 最多10000个
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f'\n{"=" * 50}')
    print(f'  法考法条库 激活码生成工具')
    print(f'  生成时间：{timestamp}')
    print(f'  生成数量：{count} 个')
    print(f'{"=" * 50}\n')

    codes = [generate_key() for _ in range(count)]

    for i, code in enumerate(codes, 1):
        print(f'  {i:>4}.  {code}')

    print(f'\n{"=" * 50}')
    print(f'  共 {count} 个激活码生成完毕')
    print(f'{"=" * 50}\n')

    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f'# 法考法条库激活码 - 生成于 {timestamp}\n')
            f.write(f'# 共 {count} 个\n\n')
            for code in codes:
                f.write(code + '\n')
        print(f'✅ 已保存到文件：{args.output}\n')


if __name__ == '__main__':
    main()
