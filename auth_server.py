import json
import os
import uuid
import hmac
import hashlib
import secrets
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# Vercel Serverless 环境中，只有 /tmp/ 目录有写权限
DB_FILE = '/tmp/users.json'
CODES_FILE = '/tmp/codes.json'

# ============================================================
# 核心密钥 —— 请勿泄露！
# ============================================================
SECRET_KEY = 'LawFaKao@2024_HMAC_K3y_#Z9!mX'

# 字符集：32个无歧义字符（去掉 0/O/1/I 防止混淆）
CHARSET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'


# ============================================================
# 激活码核心算法
# ============================================================

def _to_base32_str(n: int, length: int) -> str:
    """将整数 n 编码为 CHARSET 表示的 length 位字符串"""
    result = []
    for _ in range(length):
        result.append(CHARSET[n % 32])
        n //= 32
    return ''.join(reversed(result))


def generate_key(prefix: str = '') -> str:
    """
    生成10位激活码，格式：XXXXX-XXXXX
    - 随机载荷：7位（CHARSET随机字符）
    - 校验码：  3位（HMAC-SHA256签名的Base-32编码）
    - 最终展示：中间加"-"分隔，便于阅读和输入
    """
    payload = ''.join(secrets.choice(CHARSET) for _ in range(7))
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    n = int.from_bytes(sig[:3], 'big') % (32 ** 3)
    checksum = _to_base32_str(n, 3)
    raw = payload + checksum  # 共10位
    return f'{raw[:5]}-{raw[5:]}'  # 展示为 XXXXX-XXXXX


def validate_key(key_input: str):
    """
    校验激活码格式与签名。
    返回 (True, 标准化后的10位码) 或 (False, 错误说明)
    """
    # 标准化：去掉"-"、空格，转大写
    key = key_input.strip().upper().replace('-', '').replace(' ', '')

    if len(key) != 10:
        return False, f'激活码长度错误（应为10位，当前 {len(key)} 位）'

    for c in key:
        if c not in CHARSET:
            return False, f'含有不合法字符：{c}'

    payload = key[:7]
    checksum = key[7:]

    # 重新计算期望签名
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    n = int.from_bytes(sig[:3], 'big') % (32 ** 3)
    expected = _to_base32_str(n, 3)

    if checksum != expected:
        return False, '激活码无效，请确认后重新输入'

    return True, key


# ============================================================
# 数据库 I/O
# ============================================================

def _init_file(path, default):
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default, f)


def load_users():
    _init_file(DB_FILE, {})
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_users(users):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=4, ensure_ascii=False)


def load_codes():
    _init_file(CODES_FILE, {})
    with open(CODES_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_codes(codes):
    with open(CODES_FILE, 'w', encoding='utf-8') as f:
        json.dump(codes, f, indent=4, ensure_ascii=False)


# ============================================================
# API 路由
# ============================================================

@app.route('/register', methods=['POST'])
def register():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'code': 400, 'msg': '邮箱和密码不能为空'}), 400

    users = load_users()
    if username in users:
        return jsonify({'code': 400, 'msg': '该邮箱已被注册'}), 400

    users[username] = {
        'password_hash': generate_password_hash(password),
        'activated': False,
        'activated_at': None,
        'code_used': None
    }
    save_users(users)
    return jsonify({'code': 200, 'msg': '注册成功'})


@app.route('/login', methods=['POST'])
def login():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'code': 400, 'msg': '邮箱和密码不能为空'}), 400

    users = load_users()
    user = users.get(username)
    if user and check_password_hash(user['password_hash'], password):
        token = str(uuid.uuid4())
        return jsonify({
            'code': 200,
            'msg': '登录成功',
            'token': token,
            'activated': user.get('activated', False)
        })
    return jsonify({'code': 401, 'msg': '邮箱账号或密码错误'}), 401


@app.route('/redeem', methods=['POST'])
def redeem():
    data = request.json or {}
    username = data.get('username', '').strip()
    code_input = data.get('code', '').strip()

    if not username or not code_input:
        return jsonify({'code': 400, 'msg': '参数不完整'}), 400

    # 1. 校验格式 + HMAC 签名
    ok, result = validate_key(code_input)
    if not ok:
        return jsonify({'code': 400, 'msg': result}), 400
    code = result  # 标准化后的10位码（无"-"）

    # 2. 用户是否存在
    users = load_users()
    if username not in users:
        return jsonify({'code': 404, 'msg': '用户不存在'}), 404

    # 3. 该账号是否已激活
    if users[username].get('activated'):
        return jsonify({'code': 400, 'msg': '您的账号已激活，无需重复兑换'}), 400

    # 4. 激活码是否被使用过（一码一用）
    codes = load_codes()
    if code in codes:
        return jsonify({'code': 400, 'msg': '该激活码已被使用，请换一个'}), 400

    # 5. 写入记录，激活账号
    codes[code] = {
        'used_by': username,
        'used_at': datetime.datetime.now().isoformat()
    }
    save_codes(codes)

    users[username]['activated'] = True
    users[username]['activated_at'] = datetime.datetime.now().isoformat()
    users[username]['code_used'] = code
    save_users(users)

    return jsonify({'code': 200, 'msg': '激活成功！即刻享受完整功能'})


@app.route('/change_password', methods=['POST'])
def change_password():
    data = request.json or {}
    username = data.get('username', '').strip()
    old_pwd = data.get('old_password', '').strip()
    new_pwd = data.get('new_password', '').strip()

    if not username or not old_pwd or not new_pwd:
        return jsonify({'code': 400, 'msg': '参数不完整'}), 400

    users = load_users()
    user = users.get(username)
    if not user or not check_password_hash(user['password_hash'], old_pwd):
        return jsonify({'code': 401, 'msg': '原密码错误'}), 401

    users[username]['password_hash'] = generate_password_hash(new_pwd)
    save_users(users)
    return jsonify({'code': 200, 'msg': '密码修改成功'})


# ============================================================
# 启动
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5100))
    print('=========================================')
    print(' 法考法条库 后端认证服务器 v2.1')
    print(f' 运行在端口: {port}')
    print(' 激活码算法：HMAC-SHA256 / 10位自校验')
    print('=========================================')
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
