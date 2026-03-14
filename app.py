"""
app.py  ──  ContractClarity 后端服务（专家增强版 v4.0）

新增四大核心增强（在 v2.1 基础上）：

★ 增强 A: 多智能体辩论仲裁（Multi-Agent Debate · System 2 慢思考）
   ┌─ v3.0 并行快思考（保留）：多模型同时独立推断 → 共识合并
   └─ v4.0 辩论慢思考（新增）：对高危风险点引入三方角色对峙
        甲方律师 Agent  ── 从合同签署方视角挑战风险结论、寻找有利解释
        乙方律师 Agent  ── 从对手视角强化风险论据、援引不利判例
        仲裁官   Agent  ── 中立裁判双方论点，输出最终裁定结论
      → 仲裁官裁定覆盖并行阶段初稿，消除"幻觉"，对齐 System 2 严谨度
      → 每个辩题独立并行（issues 间并发），总耗时不超过最慢单题
      → debate_rounds / arbitrator_verdict / severity_adjustment 写入结果
   - 环境变量：ENABLE_DEBATE_MODE=true（默认 false）
               DEBATE_MAX_ROUNDS=1（辩论轮数，1-2）
               DEBATE_TOP_N_ISSUES=3（最多辩论前 N 个高危议题）
               DEBATE_TIMEOUT=150（单议题辩论超时秒数）

★ 增强 B: 动态法律知识库（结合 ingest.py v2.0）
   - 向量检索时区分文档类型：法律法规 / 司法解释 / 典型案例 / 部门规章
   - 案例型文档触发专门的"案例类比分析"子阶段
   - 知识库版本感知：读取 kb_manifest.json，在报告中展示知识库更新日期
   - 支持按 document_type 过滤的精准检索

★ 增强 C: 专家律师介入（Expert-in-the-Loop）
   - 新增 lawyers 表：注册专业律师账户（执照号、专长领域）
   - 新增 lawyer_reviews 表：人工审查任务管理（状态机）
   - 高风险合同（riskScore ≥ 75）自动建议申请人工审查
   - /review/request   ── 用户申请律师审查
   - /review/submit    ── 律师提交专业意见（需律师 JWT）
   - /review/list      ── 用户查看自己的审查记录
   - /review/<id>      ── 获取具体审查详情
   - /lawyer/register  ── 律师注册
   - /lawyer/pending   ── 律师查看待审任务
"""

import os
import json
import uuid
import hashlib
import threading
import traceback
import sqlite3
import secrets
import time
import re
import random
import queue as _queue
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import wraps

import bcrypt
import jwt as pyjwt

from dotenv import load_dotenv
from flask import Flask, request, jsonify, g, Response, stream_with_context, send_from_directory
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

RBAC_ROLES = {
    "admin": {
        "label": "管理员",
        "permissions": ["view_all", "download", "share", "manage_users", "admin_panel"]
    },
    "legal_staff": {
        "label": "法务人员",
        "permissions": ["view_all", "download", "share"]
    },
    "employee": {
        "label": "普通员工",
        "permissions": ["view_own"]
    }
}

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("未配置 DEEPSEEK_API_KEY 环境变量，请检查 .env 文件")

# ════════════════════════════════════════════════════════════════
#  ★ 增强 A 配置：多模型交叉验证
# ════════════════════════════════════════════════════════════════
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
MOONSHOT_API_KEY    = os.getenv("MOONSHOT_API_KEY", "")
DASHSCOPE_API_KEY    = os.getenv("DASHSCOPE_API_KEY", "")
GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY", "")
SILICONFLOW_API_KEY  = os.getenv("SILICONFLOW_API_KEY", "")
ZHIPU_API_KEY        = os.getenv("ZHIPU_API_KEY", "") 
ENABLE_MULTI_MODEL  = os.getenv("ENABLE_MULTI_MODEL", "false").lower() == "true"
MULTI_MODEL_TIMEOUT = int(os.getenv("MULTI_MODEL_TIMEOUT", "150"))  # 每个模型最大等待秒数
# 置信度阈值：多模型共识度低于此值时在报告中标记"存在分歧"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# ── ★ 增强 A v4.0: 辩论仲裁配置 ──────────────────────────────────
# 是否开启辩论模式（消耗更多 tokens，但显著提升高危风险点的准确性）
ENABLE_DEBATE_MODE    = os.getenv("ENABLE_DEBATE_MODE",    "false").lower() == "true"
# 辩论轮数：1=单轮立场+裁定，2=加一轮反驳
DEBATE_MAX_ROUNDS     = max(1, min(2, int(os.getenv("DEBATE_MAX_ROUNDS", "1"))))
# 最多对前 N 个高危议题展开辩论（避免费用失控）
DEBATE_TOP_N_ISSUES   = max(1, min(5, int(os.getenv("DEBATE_TOP_N_ISSUES", "2"))))
# 单个议题辩论超时（秒）
DEBATE_TIMEOUT        = int(os.getenv("DEBATE_TIMEOUT", "120"))

# ════════════════════════════════════════════════════════════════
#  ★ 增强 C 配置：专家审查
# ════════════════════════════════════════════════════════════════
# 风险分阈值：超过此值自动建议申请人工审查
AUTO_SUGGEST_REVIEW_SCORE = int(os.getenv("AUTO_SUGGEST_REVIEW_SCORE", "75"))
# 律师 JWT 类型标识
LAWYER_TOKEN_TYPE = "lawyer_access"

# ==================== 企业级鉴权配置 ====================
JWT_SECRET          = os.getenv("JWT_SECRET", secrets.token_hex(32))
DEV_MODE            = os.getenv("DEV_MODE", "true").lower() == "true"
ACCESS_TOKEN_HOURS  = int(os.getenv("ACCESS_TOKEN_HOURS",  "2"))
REFRESH_TOKEN_DAYS  = int(os.getenv("REFRESH_TOKEN_DAYS",  "30"))
OTP_EXPIRE_SECONDS  = 300
OTP_MAX_ATTEMPTS    = 5
OTP_RATE_LIMIT_SEC  = 60
DB_PATH = os.getenv("DB_PATH", "contractclarity.db")

# ==================== 分析安全与缓存配置 ====================
ANALYSIS_RATE_LIMIT_PER_HOUR = int(os.getenv("ANALYSIS_RATE_LIMIT_PER_HOUR", "20"))
CACHE_TTL_HOURS     = int(os.getenv("CACHE_TTL_HOURS",    "24"))
MIN_CONTRACT_LENGTH = 100
MAX_CONTRACT_LENGTH = int(os.getenv("MAX_CONTRACT_LENGTH", "10000"))
LLM_MAX_RETRIES     = int(os.getenv("LLM_MAX_RETRIES",    "3"))
ANALYSIS_VERSION    = "4.0"   # 多智能体辩论仲裁版本

_ip_rate_store: dict = {}
_ip_rate_lock  = threading.Lock()
_otp_store     = {}
_token_bl      = set()
_store_lock    = threading.Lock()


# ════════════════════════════════════════════════════════════════
#  数据库初始化（包含新增的 lawyers / lawyer_reviews 表）
# ════════════════════════════════════════════════════════════════
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            phone         TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            nickname      TEXT,
            email         TEXT DEFAULT '',
            bio           TEXT DEFAULT '',
            plan          TEXT DEFAULT 'free',
            role          TEXT DEFAULT 'employee',       -- 新增：角色
            department    TEXT DEFAULT '',              -- 新增：部门
            review_count  INTEGER DEFAULT 0,
            join_date     TEXT NOT NULL,
            notifications TEXT DEFAULT '{"emailNotif":true,"smsNotif":false,"weeklyReport":true,"riskAlert":true}',
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS contracts (
            id                  TEXT PRIMARY KEY,
            user_id             TEXT NOT NULL,
            date                TEXT NOT NULL,
            category            TEXT,
            contract_type       TEXT,
            risk_score          INTEGER DEFAULT 0,
            overall_risk        TEXT,
            summary             TEXT,
            jurisdiction        TEXT,
            issues              TEXT DEFAULT '[]',
            confidence_score    REAL DEFAULT NULL,
            models_used         TEXT DEFAULT NULL,
            review_requested    INTEGER DEFAULT 0,
            created_at          TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS favorites (
            user_id     TEXT NOT NULL,
            contract_id TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (user_id, contract_id),
            FOREIGN KEY (user_id)     REFERENCES users(id)     ON DELETE CASCADE,
            FOREIGN KEY (contract_id) REFERENCES contracts(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_contracts_user_id ON contracts(user_id);
        CREATE INDEX IF NOT EXISTS idx_favorites_user_id ON favorites(user_id);

        -- 分析结果缓存
        CREATE TABLE IF NOT EXISTS analysis_cache (
            hash        TEXT NOT NULL,
            category    TEXT NOT NULL,
            language    TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            hit_count   INTEGER DEFAULT 0,
            PRIMARY KEY (hash, category, language)
        );
        CREATE INDEX IF NOT EXISTS idx_cache_created ON analysis_cache(created_at);

        -- 分析审计日志
        CREATE TABLE IF NOT EXISTS analysis_audit (
            id          TEXT PRIMARY KEY,
            ip          TEXT,
            text_hash   TEXT NOT NULL,
            category    TEXT NOT NULL,
            language    TEXT NOT NULL,
            status      TEXT DEFAULT 'started',
            cached      INTEGER DEFAULT 0,
            models_used TEXT DEFAULT NULL,
            started_at  TEXT NOT NULL,
            ended_at    TEXT
        );

        -- ★ 增强 C: 注册律师账户表
        CREATE TABLE IF NOT EXISTS lawyers (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            license_number  TEXT UNIQUE NOT NULL,
            firm            TEXT,
            specialties     TEXT DEFAULT '[]',   -- JSON 数组：['劳动用工类','合同纠纷']
            phone           TEXT UNIQUE NOT NULL,
            password_hash   TEXT NOT NULL,
            verified        INTEGER DEFAULT 0,   -- 0=待核验, 1=已认证
            rating          REAL DEFAULT 5.0,
            completed_reviews INTEGER DEFAULT 0,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        -- ★ 增强 C: 律师审查任务表
        CREATE TABLE IF NOT EXISTS lawyer_reviews (
            id              TEXT PRIMARY KEY,
            contract_id     TEXT,               -- 关联合同记录（可空，合同可能未保存）
            user_id         TEXT NOT NULL,
            lawyer_id       TEXT,               -- 分配后填入
            category        TEXT NOT NULL,
            risk_score      INTEGER,
            contract_text   TEXT NOT NULL,      -- 存储合同原文（律师阅览用）
            ai_result_json  TEXT,               -- AI 初稿结果
            status          TEXT DEFAULT 'pending',
            -- 状态机：pending → assigned → in_review → completed / rejected
            priority        TEXT DEFAULT 'normal',  -- normal / urgent / critical
            lawyer_opinion  TEXT,               -- 律师专业意见
            lawyer_notes    TEXT,               -- 律师内部备注
            endorsement     TEXT DEFAULT 'none',
            -- none / endorsed（律师认可AI结论）/ amended（律师修改了结论）/ overridden（律师推翻AI结论）
            endorsed_result_json TEXT,          -- 律师认可/修订后的最终结果
            user_notes      TEXT,               -- 用户提交时的备注
            completed_at    TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            FOREIGN KEY (user_id)   REFERENCES users(id)   ON DELETE CASCADE,
            FOREIGN KEY (lawyer_id) REFERENCES lawyers(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reviews_user_id   ON lawyer_reviews(user_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_lawyer_id ON lawyer_reviews(lawyer_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_status    ON lawyer_reviews(status);
    """)
    conn.commit() # 必须先提交创建表的指令

    # 2. 然后执行迁移逻辑
    try:
        # 添加角色和部门列
        cursor = conn.execute("PRAGMA table_info(users)")
        cols = [column[1] for column in cursor.fetchall()]
        if "role" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'employee'")
        if "department" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN department TEXT DEFAULT ''")
        conn.commit()
        
        # 自动设第一个注册者为 admin
        admin_exists = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        if admin_exists == 0:
            conn.execute("UPDATE users SET role='admin' WHERE id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1)")
            conn.commit()
    except Exception as e:
        print(f"数据库升级提醒: {e}")
    finally:
        conn.close()
    print("数据库初始化完成:", DB_PATH)

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, 
            static_folder=os.path.join(basedir, 'static'),
            static_url_path='/static')
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "bypass-tunnel-reminder", "ngrok-skip-browser-warning"]
}})

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def send_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.before_request
def handle_options_preflight():
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = '*'
        return res
    

def _get_user_effective_permissions(conn, user_id: str) -> list:
    row = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        return []
    role = row["role"] or "employee"
    # 获取 RBAC_ROLES 中定义的权限列表
    return RBAC_ROLES.get(role, {}).get("permissions", [])

def require_permission(perm):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            db = get_db()
            perms = _get_user_effective_permissions(db, g.user_id)
            if perm not in perms:
                return jsonify({"error": "forbidden", "message": f"权限不足，需要 {perm}"}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ==================== 鉴权中间件 ====================
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "missing_token", "message": "需要登录"}), 401
        token = auth_header[7:]
        try:
            payload = verify_token(token, "access")
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "token_expired", "message": "登录已过期，请刷新令牌"}), 401
        except pyjwt.InvalidTokenError as e:
            return jsonify({"error": "invalid_token", "message": str(e)}), 401
        g.user_id = payload["sub"]
        return f(*args, **kwargs)
    return decorated


def require_lawyer_auth(f):
    """律师专用路由装饰器：验证律师 JWT 并注入 g.lawyer_id"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "missing_token", "message": "需要律师账号登录"}), 401
        token = auth_header[7:]
        try:
            payload = verify_token(token, LAWYER_TOKEN_TYPE)
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "token_expired", "message": "登录已过期"}), 401
        except pyjwt.InvalidTokenError as e:
            return jsonify({"error": "invalid_token", "message": str(e)}), 401
        g.lawyer_id = payload["sub"]
        return f(*args, **kwargs)
    return decorated

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3", 
    openai_api_key=os.getenv("SILICONFLOW_API_KEY"), 
    openai_api_base="https://api.siliconflow.cn/v1"
)

@app.route('/auth/profile', methods=['PUT'])
@require_auth
def auth_update_profile():
    return auth_update_me() # 复用现有的更新逻辑

@app.route('/auth/notifications', methods=['PUT'])
@require_auth
def auth_update_notifications():
    return auth_update_me() # 复用现有的更新逻辑

@app.route('/auth/me/permissions', methods=['GET'])
@require_auth
def auth_my_permissions():
    """返回当前用户的角色和详细权限清单清单"""
    db = get_db()
    user = db.execute("SELECT role FROM users WHERE id=?", (g.user_id,)).fetchone()
    role = user['role'] if user else 'employee'
    
    role_info = RBAC_ROLES.get(role, {"label": "普通用户", "permissions": []})
    
    return jsonify({
        "role": role,
        "role_label": role_info["label"],
        "permissions": role_info["permissions"],
        "all_roles": RBAC_ROLES # 必须把整个字典传给前端，前端需要据此渲染角色颜色
    })

@app.route('/admin/users', methods=['GET'])
@require_auth
@require_permission('admin_panel') # 使用装饰器代替硬编码判断
def admin_list_users():
    db = get_db()
    search = request.args.get('search', '')
    query = "SELECT id, phone, nickname, role, created_at FROM users"
    params = []
    if search:
        query += " WHERE phone LIKE ? OR nickname LIKE ?"
        params = [f"%{search}%", f"%{search}%"]
    
    rows = db.execute(query, params).fetchall()
    
    users_out = []
    for r in rows:
        u = dict(r)
        # 为每个用户注入其权限清单，方便前端展示
        u["permissions"] = RBAC_ROLES.get(u["role"], {}).get("permissions", [])
        users_out.append(u)
        
    return jsonify({"users": users_out})

@app.route('/admin/users/<uid>/role', methods=['PUT'])
@require_auth
def admin_update_role(uid):
    """管理后台：修改用户角色"""
    db = get_db()
    me = db.execute("SELECT role FROM users WHERE id=?", (g.user_id,)).fetchone()
    if me['role'] != 'admin':
        return jsonify({"error": "拒绝访问"}), 403

    new_role = request.json.get('role')
    if new_role not in RBAC_ROLES:
        return jsonify({"error": "角色不存在"}), 400

    db.execute("UPDATE users SET role=? WHERE id=?", (new_role, uid))
    db.commit()
    return jsonify({"message": "更新成功"})

# ==================== JWT 工具函数 ====================
def _now_ts():
    return int(datetime.now(timezone.utc).timestamp())


def issue_access_token(user_id: str) -> str:
    jti = secrets.token_hex(16)
    payload = {
        "sub":  user_id,
        "jti":  jti,
        "type": "access",
        "iat":  _now_ts(),
        "exp":  _now_ts() + ACCESS_TOKEN_HOURS * 3600,
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")


def issue_refresh_token(user_id: str) -> str:
    jti = secrets.token_hex(16)
    payload = {
        "sub":  user_id,
        "jti":  jti,
        "type": "refresh",
        "iat":  _now_ts(),
        "exp":  _now_ts() + REFRESH_TOKEN_DAYS * 86400,
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")


def issue_lawyer_token(lawyer_id: str) -> str:
    """为认证律师签发特殊 JWT（type=lawyer_access）"""
    jti = secrets.token_hex(16)
    payload = {
        "sub":  lawyer_id,
        "jti":  jti,
        "type": LAWYER_TOKEN_TYPE,
        "iat":  _now_ts(),
        "exp":  _now_ts() + ACCESS_TOKEN_HOURS * 3600,
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str, expected_type: str = "access"):
    payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    if payload.get("type") != expected_type:
        raise pyjwt.InvalidTokenError("token type mismatch")
    with _store_lock:
        if payload.get("jti") in _token_bl:
            raise pyjwt.InvalidTokenError("token has been revoked")
    return payload


def revoke_token(token: str):
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        with _store_lock:
            _token_bl.add(payload.get("jti"))
    except Exception:
        pass

# ==================== OTP 工具函数 ====================
def _otp_generate(phone: str) -> tuple[str, str]:
    now = _now_ts()
    with _store_lock:
        existing = _otp_store.get(phone)
        if existing and now - existing.get("issued_at", 0) < OTP_RATE_LIMIT_SEC:
            remaining = OTP_RATE_LIMIT_SEC - (now - existing["issued_at"])
            raise ValueError(f"发送过于频繁，请 {remaining} 秒后重试")
        code = f"{random.randint(0, 999999):06d}"
        _otp_store[phone] = {
            "code":      code,
            "expiry":    now + OTP_EXPIRE_SECONDS,
            "attempts":  0,
            "issued_at": now,
        }
    print(f"[OTP] phone={phone}  code={code}  (dev_mode={DEV_MODE})")
    dev_hint = code if DEV_MODE else ""
    return code, dev_hint


def _otp_verify(phone: str, code: str) -> bool:
    now = _now_ts()
    with _store_lock:
        record = _otp_store.get(phone)
        if not record:
            raise ValueError("验证码不存在或已过期，请重新获取")
        if now > record["expiry"]:
            del _otp_store[phone]
            raise ValueError("验证码已过期，请重新获取")
        record["attempts"] += 1
        if record["attempts"] > OTP_MAX_ATTEMPTS:
            del _otp_store[phone]
            raise ValueError("验证码错误次数过多，请重新获取")
        if record["code"] != code:
            remaining = OTP_MAX_ATTEMPTS - record["attempts"]
            raise ValueError(f"验证码错误，还可尝试 {remaining} 次")
        del _otp_store[phone]
    return True


# ==================== 用户数据库操作 ====================
def _user_to_dict(row) -> dict:
    if not row:
        return None
    d = dict(row)
    d.pop("password_hash", None)
    # 确保以下字段一定存在（即便数据库里是空的）
    d["role"] = d.get("role") or "employee" 
    d["nickname"] = d.get("nickname") or "未定义"
    try:
        d["notifications"] = json.loads(d.get("notifications") or "{}")
    except Exception:
        d["notifications"] = {}
    return d


def _db_get_user_by_id(conn, user_id: str):
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _user_to_dict(row)


def _db_get_user_by_phone(conn, phone: str):
    row = conn.execute("SELECT * FROM users WHERE phone=?", (phone,)).fetchone()
    return dict(row) if row else None


def _db_create_user(conn, phone: str, password_hash=None, nickname=None) -> dict:
    now_str = datetime.now(timezone.utc).isoformat()
    uid = str(uuid.uuid4())
    nickname = nickname or (phone[:3] + "****" + phone[-4:])
    conn.execute(
        """INSERT INTO users (id, phone, password_hash, nickname, join_date, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (uid, phone, password_hash, nickname, now_str, now_str, now_str)
    )
    conn.commit()
    return _db_get_user_by_id(conn, uid)


# ==================== 合同分析工具函数 ====================
def _contract_hash(text: str, category: str, language: str, party_role: str = 'partyA') -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(f"{normalized}|{category}|{language}|{party_role}".encode()).hexdigest()


def _retry_llm(llm_call, max_attempts: int = LLM_MAX_RETRIES, initial_wait: float = 2.0):
    for attempt in range(max_attempts):
        try:
            return llm_call()
        except Exception as e:
            if attempt >= max_attempts - 1:
                raise
            wait = initial_wait * (2 ** attempt)
            print(f"  ↺ LLM 调用失败（第 {attempt+1}/{max_attempts} 次）：{e}，{wait:.1f}s 后重试...")
            time.sleep(wait)


def _validate_and_repair_schema(data: dict) -> dict:
    required_top = ['contractType', 'jurisdiction', 'overallRisk', 'riskScore', 'summary', 'issues']
    for field in required_top:
        if field not in data:
            if field == 'issues':
                data[field] = []
            elif field == 'riskScore':
                data[field] = 50
            else:
                data[field] = f'[{field} 未提取]'
            print(f"  ⚠ Schema 修复：补充缺失字段 '{field}'")

    if not isinstance(data['issues'], list):
        raise ValueError(f"'issues' 字段类型错误，期望 list，实际 {type(data['issues'])}")

    try:
        data['riskScore'] = max(0, min(100, int(float(str(data['riskScore'])))))
    except (ValueError, TypeError):
        data['riskScore'] = 50

    issue_required = {
        'id': 0, 'severity': '中', 'title': '未识别风险', 'problem': '—',
        'clauseText': '', 'lawReference': '—', 'plainLanguage': [], 'whatToDo': [], 'alternative': '—'
    }
    for i, issue in enumerate(data['issues']):
        if not isinstance(issue, dict):
            data['issues'][i] = {'id': i + 1, 'severity': '中', 'title': f'风险点 {i+1}', 'problem': str(issue)}
            continue
        for f, default in issue_required.items():
            if f not in issue:
                issue[f] = default
        for list_field in ('plainLanguage', 'whatToDo'):
            if isinstance(issue.get(list_field), str):
                issue[list_field] = [issue[list_field]]

    return data


def _sanitize_contract_text(text: str) -> tuple[str, list]:
    warnings = []
    cleaned = re.sub(r'<[^>]{1,200}>', '', text)
    cleaned = re.sub(r'\n{5,}', '\n\n\n', cleaned)
    injection_patterns = [
        (re.compile(r'ignore\s+(previous|all|above)\s+(instructions|prompts)', re.I), '[已过滤]'),
        (re.compile(r'(system|new)\s+prompt[\s:：]', re.I),                          '[已过滤]'),
        (re.compile(r'你(现在|从现在起|即将)?(扮演|是|变成)', re.I),                   '[已过滤]'),
    ]
    for pat, replacement in injection_patterns:
        if pat.search(cleaned):
            warnings.append('检测到可疑内容模式，相关片段已过滤')
            cleaned = pat.sub(replacement, cleaned)

    cleaned = cleaned.strip()
    if len(cleaned) < MIN_CONTRACT_LENGTH:
        raise ValueError(f"合同内容不足 {MIN_CONTRACT_LENGTH} 字（当前 {len(cleaned)} 字），请提供更完整的合同文本")
    if len(cleaned) > MAX_CONTRACT_LENGTH:
        cleaned = cleaned[:MAX_CONTRACT_LENGTH]
        warnings.append(f'合同内容超出 {MAX_CONTRACT_LENGTH} 字限制，已自动截断至最大值')

    return cleaned, warnings


def _check_analysis_rate_limit(ip: str) -> tuple[bool, int]:
    now = time.time()
    window = 3600
    with _ip_rate_lock:
        ts_list = [t for t in _ip_rate_store.get(ip, []) if now - t < window]
        if len(ts_list) >= ANALYSIS_RATE_LIMIT_PER_HOUR:
            _ip_rate_store[ip] = ts_list
            return False, 0
        ts_list.append(now)
        _ip_rate_store[ip] = ts_list
        return True, ANALYSIS_RATE_LIMIT_PER_HOUR - len(ts_list)


def _get_cached_result(text_hash: str, category: str, language: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS)).isoformat()
        row = conn.execute(
            "SELECT result_json, hit_count FROM analysis_cache "
            "WHERE hash=? AND category=? AND language=? AND created_at > ?",
            (text_hash, category, language, cutoff)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE analysis_cache SET hit_count=hit_count+1 "
                "WHERE hash=? AND category=? AND language=?",
                (text_hash, category, language)
            )
            conn.commit()
            result = json.loads(row['result_json'])
            result['_cached'] = True
            result['_cache_hits'] = (row['hit_count'] or 0) + 1
            return result
    except Exception as e:
        print(f"缓存查询错误: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return None


def _save_to_cache(text_hash: str, category: str, language: str, result: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        now_str = datetime.now(timezone.utc).isoformat()
        to_cache = {k: v for k, v in result.items() if not k.startswith('_')}
        conn.execute(
            "INSERT OR REPLACE INTO analysis_cache "
            "(hash, category, language, result_json, created_at, hit_count) "
            "VALUES (?,?,?,?,?,0)",
            (text_hash, category, language, json.dumps(to_cache, ensure_ascii=False), now_str)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"缓存写入错误: {e}")


def _log_analysis_audit(audit_id: str, ip: str, text_hash: str,
                         category: str, language: str,
                         status: str = 'started', cached: bool = False,
                         ended: bool = False, models_used: list = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        now_str = datetime.now(timezone.utc).isoformat()
        if status == 'started':
            conn.execute(
                "INSERT OR IGNORE INTO analysis_audit "
                "(id, ip, text_hash, category, language, status, cached, started_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (audit_id, ip, text_hash, category, language, status, int(cached), now_str)
            )
        else:
            models_json = json.dumps(models_used) if models_used else None
            conn.execute(
                "UPDATE analysis_audit SET status=?, ended_at=?, models_used=? WHERE id=?",
                (status, now_str if ended else None, models_json, audit_id)
            )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
#  ★ 增强 A: 多模型管理器
# ════════════════════════════════════════════════════════════════

def _get_available_models() -> list[dict]:
    """
    返回当前可用的 LLM 配置列表。
    每个元素：{"name": "...", "llm": ChatOpenAI实例, "weight": 1.0}
    """
    models = []

    # 主力模型：DeepSeek（始终可用）
    models.append({
        "name":   "deepseek-chat",
        "weight": 1.0,
        "llm": ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com",
            max_tokens=3000,
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    })
    
    if not ENABLE_MULTI_MODEL:
        return models

    print(f"\n[多模型诊断] 正在组建专家审查团...")

    # 2. 阿里云 (通义千问 Qwen-Max)
    if DASHSCOPE_API_KEY:
        print(f"[多模型诊断] 加入：阿里云 Qwen-Max")
        models.append({
            "name": "qwen-max",
            "weight": 0.95,
            "llm": ChatOpenAI(
                model='qwen-max',
                openai_api_key=DASHSCOPE_API_KEY,
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                max_tokens=3000,
                temperature=0.2,
            )
        })

    # 5. Moonshot (Kimi)
    if MOONSHOT_API_KEY:
        print(f"[多模型诊断] 加入：月之暗面 Kimi")
        models.append({
            "name": "moonshot-v1-8k",
            "weight": 0.9,
            "llm": ChatOpenAI(
                model='moonshot-v1-8k',
                openai_api_key=MOONSHOT_API_KEY,
                openai_api_base="https://api.moonshot.cn/v1",
                max_tokens=3000,
                temperature=0.2,
            )
        })

    # 6. 智谱 AI (GLM-4)
    if ZHIPU_API_KEY:
        print(f"[多模型诊断] 加入：智谱 AI GLM-4")
        models.append({
            "name": "glm-4",
            "weight": 0.9,
            "llm": ChatOpenAI(
                model='glm-4',
                openai_api_key=ZHIPU_API_KEY,
                openai_api_base="https://open.bigmodel.cn/api/paas/v4",
                max_tokens=3000,
                temperature=0.2,
            )
        })

    return models


def _run_single_model_review(model_cfg: dict, prompt: str) -> dict | None:
    """
    在单个模型上执行风险审查。
    返回解析后的 JSON dict，失败时返回 None。
    """
    model_name = model_cfg["name"]
    llm = model_cfg["llm"]
    try:
        res = _retry_llm(lambda: llm.invoke(prompt))
        data = json.loads(robust_json_cleaner(res.content), strict=False)
        data = _validate_and_repair_schema(data)
        data["_model_name"] = model_name
        print(f"   ✅ [{model_name}] 风险审查完成，识别风险点 {len(data.get('issues',[]))} 个")
        return data
    except Exception as e:
        print(f"   ⚠ [{model_name}] 审查失败：{e}")
        return None


def _normalize_severity(s: str) -> str:
    """统一不同模型可能输出的风险等级表示"""
    s = str(s).strip()
    mapping = {
        '极高': '极高', 'very high': '极高', 'critical': '极高', '严重': '极高',
        '高':   '高',   'high': '高',
        '中':   '中',   'medium': '中', 'moderate': '中',
        '低':   '低',   'low': '低',
    }
    return mapping.get(s.lower(), s)


def _title_similarity(a: str, b: str) -> float:
    """
    简单的标题相似度（基于词集合交集）。
    用于跨模型风险点聚类，不依赖外部库。
    """
    a_words = set(re.sub(r'[^\w]', '', a).lower())
    b_words = set(re.sub(r'[^\w]', '', b).lower())
    if not a_words or not b_words:
        return 0.0
    intersection = len(a_words & b_words)
    union = len(a_words | b_words)
    return intersection / union if union > 0 else 0.0


def _merge_multi_model_results(model_results: list[dict], model_weights: list[float]) -> dict:
    """
    ★ 核心共识算法：合并多模型风险审查结果。（修复置信度溢出版本）
    """
    # 过滤掉失败的模型结果
    valid = [(r, w) for r, w in zip(model_results, model_weights) if r is not None]
    if not valid:
        raise ValueError("所有模型均返回失败，无法生成分析结果")

    # 计算参与本次分析的所有模型的总权重（每个模型仅算一次）
    total_weight = sum(w for _, w in valid)

    if len(valid) == 1:
        # 单模型降级：直接返回，置信度标记为固定值
        result = valid[0][0]
        for issue in result.get('issues', []):
            issue['confidence_score']  = 0.60
            issue['model_agreement']   = '仅单模型'
            issue['models_flagged']    = [result.get('_model_name', '未知')]
        result['_multi_model'] = False
        result['_models_used'] = [result.get('_model_name')]
        return result

    # ── 1. 加权平均风险分 ──
    avg_risk_score = round(
        sum(r.get('riskScore', 50) * w for r, w in valid) / total_weight
    )

    # ── 2. 以第一个模型结果为基础框架 ──
    base_result = valid[0][0].copy()
    base_result['riskScore'] = avg_risk_score

    # ── 3. 风险等级依据平均分重新计算 ──
    if avg_risk_score >= 75:
        base_result['overallRisk'] = '极高'
    elif avg_risk_score >= 55:
        base_result['overallRisk'] = '高'
    elif avg_risk_score >= 35:
        base_result['overallRisk'] = '中'
    else:
        base_result['overallRisk'] = '低'

    # ── 4. 风险点跨模型聚类 ──
    # clusters 结构：[{"issues": [], "model_map": {model_name: weight}}]
    clusters: list[dict] = []

    for result, weight in valid:
        model_name = result.get('_model_name', '未知')
        for issue in result.get('issues', []):
            title = issue.get('title', '')
            matched = False
            
            # 尝试并入已有簇
            for cluster in clusters:
                # 使用簇内第一个问题的标题作为参考
                ref_title = cluster['issues'][0].get('title', '')
                # 提高相似度阈值到 0.45 更加精准
                if _title_similarity(title, ref_title) > 0.45:
                    cluster['issues'].append(issue)
                    # 重要修复：使用字典存储模型权重，确保同一个模型在同一个簇里只计算一次最高权重
                    cluster['model_map'][model_name] = max(cluster['model_map'].get(model_name, 0), weight)
                    matched = True
                    break
            
            if not matched:
                clusters.append({
                    'issues': [issue],
                    'model_map': {model_name: weight}
                })

    # ── 5. 为每个簇计算置信度，选出代表性 issue ──
    n_total_models = len(valid)
    merged_issues = []
    
    for idx, cluster in enumerate(clusters):
        # 选严重程度最高的 issue 作代表
        severity_order = {'极高': 4, '高': 3, '中': 2, '低': 1}
        rep_issue = max(
            cluster['issues'],
            key=lambda x: severity_order.get(_normalize_severity(x.get('severity', '中')), 2)
        )

        # 核心逻辑修复：置信度 = 该簇内唯一模型权重之和 / 总体模型总权重
        cluster_unique_weight = sum(cluster['model_map'].values())
        confidence = round(cluster_unique_weight / total_weight, 2)
        # 安全封顶
        confidence = min(1.0, confidence)

        # 多数一致性判断
        agreement_count = len(cluster['model_map'])
        if agreement_count == n_total_models:
            agreement_label = '全部一致'
        elif agreement_count >= n_total_models / 2:
            agreement_label = '多数认同'
        else:
            agreement_label = '仅单模型'

        new_issue = rep_issue.copy()
        new_issue['id']               = idx + 1
        new_issue['confidence_score'] = confidence
        new_issue['model_agreement']  = agreement_label
        new_issue['models_flagged']   = list(cluster['model_map'].keys())
        new_issue['severity']         = _normalize_severity(new_issue.get('severity', '中'))

        merged_issues.append(new_issue)

    # ── 6. 按置信度 × 严重程度排序 ──
    merged_issues.sort(
        key=lambda x: (
            x['confidence_score'],
            severity_order.get(x.get('severity', '中'), 2)
        ),
        reverse=True
    )

    # 保留前 8 个最重要的风险点
    base_result['issues']       = merged_issues[:8]
    base_result['_multi_model'] = True
    base_result['_models_used'] = [r.get('_model_name') for r, _ in valid]

    # 整体报告置信度 = 被多数模型认同（或置信度高）的风险点占比
    high_conf_count = sum(1 for i in merged_issues if i.get('confidence_score', 0) >= 0.5)
    base_result['_overall_confidence'] = round(high_conf_count / max(len(merged_issues), 1), 2)

    print(f"   🔀 合并完成：共 {len(clusters)} 簇，置信度区间: {merged_issues[-1]['confidence_score'] if merged_issues else 0} - {merged_issues[0]['confidence_score'] if merged_issues else 0}")

    return base_result

def _multi_model_risk_review(prompt: str) -> tuple[dict, list[str]]:
    """
    ★ 审查团并行调度器：并行调用所有可用模型，增加容错逻辑。
    """
    model_configs = _get_available_models()
    print(f"\n  🧑‍⚖️ 组建审查团：{[m['name'] for m in model_configs]}")

    model_results = [None] * len(model_configs)
    model_weights = [m['weight'] for m in model_configs]

    if ENABLE_MULTI_MODEL and len(model_configs) > 1:
        # 并行执行
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(_run_single_model_review, cfg, prompt): i
                for i, cfg in enumerate(model_configs)
            }
            
            try:
                # 尝试在规定时间内获取结果
                for future in as_completed(futures, timeout=MULTI_MODEL_TIMEOUT):
                    idx = futures[future]
                    try:
                        model_results[idx] = future.result()
                    except Exception as e:
                        print(f"   ✗ 模型 [{model_configs[idx]['name']}] 执行异常：{e}")
            
            except FuturesTimeoutError:
                # ★ 核心修复点：如果发生超时，不抛出异常，而是记录警告并继续
                print(f"\n  ⚠️ 部分模型审查超时（>{MULTI_MODEL_TIMEOUT}s），正在整合已完成的模型结果...")
                # 找出哪些没完成
                for f, idx in futures.items():
                    if not f.done():
                        print(f"   ⏳ 放弃超时模型: [{model_configs[idx]['name']}]")
                        # 尝试取消（虽然线程池不一定能真正杀死线程，但能防止主流程阻塞）
                        f.cancel() 
    else:
        # 单模型顺序执行
        model_results[0] = _run_single_model_review(model_configs[0], prompt)

    # 过滤掉 None，看看还剩几个
    valid_results = [r for r in model_results if r is not None]
    if not valid_results:
        raise ValueError("审查团全军覆没：所有模型均超时或调用失败，请检查网络或 API Key。")

    # 进行合并
    merged = _merge_multi_model_results(model_results, model_weights)
    models_used = merged.get('_models_used', ["未知"])
    
    return merged, models_used

# ════════════════════════════════════════════════════════════════
#  ★ 增强 A v4.0: 多智能体辩论仲裁引擎
#
#  三方角色架构：
#   甲方律师（counsel_a）── 偏向签署方，挑战风险结论
#   乙方律师（counsel_b）── 偏向对手方，强化风险论据
#   仲裁官  （arbitrator）── 中立裁判，输出最终裁定
#
#  模型分配策略（按可用性自动降级）：
#   充足（≥3 模型）→ 三方各用独立模型
#   中等（2 模型） → 甲乙各一，仲裁复用主模型（升高 temperature）
#   不足（1 模型） → 同一模型担任三种角色（不同 system prompt + temperature）
# ════════════════════════════════════════════════════════════════

def _assign_debate_roles(model_configs: list[dict]) -> dict:
    """
    ★ 精准分配角色：
    仲裁官 = DeepSeek (deepseek-chat)
    甲方律师 = Kimi (moonshot-v1-8k)
    乙方律师 = 智谱 (glm-4)
    """
    n = len(model_configs)
    if n == 0:
        raise ValueError("无可用模型，无法组建辩论庭")

    # 定义克隆工具函数
    def _mk_variant(base_cfg: dict, temperature: float, max_tokens: int = 2000) -> dict:
        orig = base_cfg["llm"]
        if hasattr(orig, 'model_name'):
            m_name = orig.model_name
        elif hasattr(orig, 'model'):
            m_name = orig.model
        else:
            m_name = base_cfg.get("name", "unknown")

        if "ChatGoogleGenerativeAI" in str(type(orig)):
            from langchain_google_genai import ChatGoogleGenerativeAI
            new_llm = ChatGoogleGenerativeAI(
                model=m_name,
                google_api_key=orig.google_api_key,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            new_llm = ChatOpenAI(
                model=m_name,
                openai_api_key=orig.openai_api_key,
                openai_api_base=orig.openai_api_base,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return {"name": base_cfg["name"], "llm": new_llm}

    # 1. 将可用模型转为字典方便查找
    cfg_map = {cfg["name"]: cfg for cfg in model_configs}

    # 2. 尝试获取指定模型，如果不存在则按索引顺序拿模型兜底
    # 甲方律师优先选 Kimi
    counsel_a_base = cfg_map.get("moonshot-v1-8k") or model_configs[0]
    
    # 乙方律师优先选 智谱
    counsel_b_base = cfg_map.get("glm-4") or (model_configs[1] if n > 1 else model_configs[0])
    
    # 仲裁官优先选 DeepSeek
    arbitrator_base = cfg_map.get("deepseek-chat") or model_configs[0]

    print(f"\n  ⚖️ 辩论庭角色指定：")
    print(f"     [甲方律师] -> {counsel_a_base['name']}")
    print(f"     [乙方律师] -> {counsel_b_base['name']}")
    print(f"     [仲裁官]   -> {arbitrator_base['name']} (DeepSeek)")

    # 3. 返回分配好的角色（带上不同的温度参数）
    return {
        "counsel_a":  _mk_variant(counsel_a_base,  temperature=0.5), # 甲方律师稍微激进点
        "counsel_b":  _mk_variant(counsel_b_base,  temperature=0.5), # 乙方律师稍微激进点
        "arbitrator": _mk_variant(arbitrator_base, temperature=0.1), # 仲裁官必须极度冷静
    }


def _debate_single_issue(
    issue: dict,
    contract_text: str,
    laws_context: str,
    roles: dict,
    category: str,
    max_rounds: int = 1,
    lang_name: str = "Simplified Chinese (简体中文)",
    party_role: str = "partyA",
) -> dict:
    """
    对单个风险点执行完整的辩论→仲裁流程。
    party_role: 'partyA'（甲方）或 'partyB'（乙方），用于在辩论中明确委托方立场。

    返回的 issue dict 附加以下字段：
      debate_rounds     : list[{round, counsel_a_arg, counsel_b_arg}]
      arbitrator_verdict: {model, upheld_severity, severity_adjustment,
                           final_reasoning, key_points_upheld,
                           key_points_rejected, consensus_whatToDo}
      debate_consensus  : "全部裁定" | "部分争议" | "严重分歧"
      confidence_score  : 仲裁后置信度（覆盖并行阶段原值）
    """
    issue_title   = issue.get("title", "未命名风险")
    clause_text   = issue.get("clauseText", "")
    original_sev  = issue.get("severity", "中")
    law_ref       = issue.get("lawReference", "")
    problem_desc  = issue.get("problem", "")

    # ── 根据委托方角色明确甲乙方律师立场 ──
    user_party_label   = "甲方" if party_role == "partyA" else "乙方"
    oppose_party_label = "乙方" if party_role == "partyA" else "甲方"
    counsel_a_stance = (
        f"甲方代理律师（代表合同{user_party_label}，即本次审查的委托方）"
        if party_role == "partyA"
        else f"甲方代理律师（代表合同甲方，即本次审查中的对方当事人）"
    )
    counsel_b_stance = (
        f"乙方代理律师（代表合同乙方，即本次审查中的对方当事人）"
        if party_role == "partyA"
        else f"乙方代理律师（代表合同{user_party_label}，即本次审查的委托方）"
    )

    debate_rounds: list[dict] = []
    round_history = ""   # 累积辩论历史，供反驳轮使用

    severity_order = {"极高": 4, "高": 3, "中": 2, "低": 1}

    # ── 工具：安全调用单角色 LLM ──────────────────────────────
    def _call_role(role_cfg: dict, prompt: str, role_label: str) -> str:
        try:
            res = _retry_llm(lambda: role_cfg["llm"].invoke(prompt), max_attempts=2)
            text = res.content.strip()
            print(f"     [{role_cfg['name']}·{role_label}] 论述完成 ({len(text)} chars)")
            return text
        except Exception as e:
            print(f"     ⚠ [{role_label}] 调用失败: {e}")
            return f"[{role_label} 无法作答: {e}]"

    # ════════════════════════════════════════════
    #  辩论轮次（1～max_rounds）
    # ════════════════════════════════════════════
    for rnd in range(1, max_rounds + 1):
        round_context = (
            f"\n\n【前序辩论记录】\n{round_history}" if round_history else ""
        )
        rebuttal_note = (
            "本轮为反驳轮，请直接针对对方上轮论点逐一反驳，并补充新证据。"
            if rnd > 1 else ""
        )

        # ── 甲方律师 prompt ───────────────────────────────────
        prompt_a = f"""你是一位经验丰富的{counsel_a_stance}，正在为委托方提供辩护。
你的角色：【挑战者】——寻找有利于甲方的法律解释，质疑对手对该风险点的认定。
（本次审查的委托方为合同【{user_party_label}】，请在辩护中优先维护{user_party_label}的利益。）

【合同类型】：{category}
【待辩议题】：{issue_title}
【涉及条款原文】：{clause_text[:500] if clause_text else "（未摘录）"}
【对手风险描述】：{problem_desc[:400] if problem_desc else issue_title}
【引用法条】：{law_ref[:300] if law_ref else "（未指定）"}
{round_context}

【你的论证任务】（{rebuttal_note if rnd > 1 else "首轮立场陈述"}）：
1. 质疑该风险点的严重性或成立前提
2. 提出对甲方有利的法律解释或业务实践
3. 如有必要，援引支持己方的法条或判例
4. 明确表达你对该风险等级的判断（极高/高/中/低）及理由

请用不超过300字的专业中文论述，逻辑清晰、有理有据。不要输出JSON。"""

        # ── 乙方律师 prompt ───────────────────────────────────
        prompt_b = f"""你是一位经验丰富的{counsel_b_stance}，代表合同另一方保护其权益。
你的角色：【强化方】——深挖该风险点对对手方（非甲方）的威胁，放大潜在损失。
（本次审查的委托方为合同【{user_party_label}】，请从{oppose_party_label}的视角出发提供制衡论点。）

【合同类型】：{category}
【待辩议题】：{issue_title}
【涉及条款原文】：{clause_text[:500] if clause_text else "（未摘录）"}
【初步风险描述】：{problem_desc[:400] if problem_desc else issue_title}
【法律依据】：{law_ref[:300] if law_ref else "（未指定）"}
【参考权威法条】：
{laws_context[:1200] if laws_context else "（无）"}
{round_context}

【你的论证任务】（{rebuttal_note if rnd > 1 else "首轮立场陈述"}）：
1. 援引具体法条证明该条款违法或存在重大隐患
2. 列举实践中类似条款导致的损失场景
3. 强调若不修改该条款的实际法律后果
4. 明确表达你对该风险等级的判断（极高/高/中/低）及理由

请用不超过300字的专业中文论述，逻辑清晰、有理有据。不要输出JSON。"""

        # 并行调用甲乙双方
        a_arg, b_arg = None, None
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(_call_role, roles["counsel_a"], prompt_a, f"甲方律师·R{rnd}")
            fut_b = ex.submit(_call_role, roles["counsel_b"], prompt_b, f"乙方律师·R{rnd}")
            a_arg = fut_a.result()
            b_arg = fut_b.result()

        round_history += (
            f"\n——第{rnd}轮——\n"
            f"甲方律师：{a_arg}\n"
            f"乙方律师：{b_arg}\n"
        )
        debate_rounds.append({
            "round":         rnd,
            "counsel_a_arg": a_arg,
            "counsel_b_arg": b_arg,
        })

    # ════════════════════════════════════════════
    #  仲裁官裁定
    # ════════════════════════════════════════════
    prompt_arb = f"""你是一位资深法官和仲裁专家，正在对以下合同风险点做出最终裁定。
你的原则：客观中立、以法律事实为准绳，不偏向任何一方。

【合同类型】：{category}
【待裁风险点】：{issue_title}
【涉及条款原文】：{clause_text[:600] if clause_text else "（未摘录）"}
【初始AI风险等级】：{original_sev}
【初始AI问题描述】：{problem_desc[:400] if problem_desc else "（未提供）"}
【权威法条依据】：
{laws_context[:1500] if laws_context else "（无）"}

【双方完整辩论记录】：
{round_history}

【你的裁定任务】：
请综合以上所有信息，输出严格符合格式的 JSON 裁定结论：

{{
  "upheld_severity": "极高/高/中/低",
  "severity_adjustment": "升级/降级/维持",
  "adjustment_reason": "一句话说明为何调整或维持风险等级",
  "final_reasoning": "仲裁官综合推理（≤150字）：综合双方论点，说明最终定性理由",
  "key_points_upheld": ["认可的关键论点（≤3条）"],
  "key_points_rejected": ["驳回的关键论点（≤3条）"],
  "consensus_whatToDo": ["最终行动建议（≤3条，比原始建议更精准）"],
  "debate_consensus": "全部裁定/部分争议/严重分歧",
  "arbitrator_confidence": 0.0至1.0之间的数字
}}

注意：debate_consensus 含义——
  全部裁定 = 仲裁官明确认定，双方论点大致一致
  部分争议 = 存在1-2个实质分歧，仲裁官做出取舍
  严重分歧 = 双方立场对立，仲裁官大幅修订了原结论"""

    arb_raw = _call_role(roles["arbitrator"], prompt_arb, "仲裁官")
    
    # 解析仲裁结果
    try:
        arb_data = json.loads(robust_json_cleaner(arb_raw), strict=False)
    except Exception:
        # 解析失败时构造保守默认值
        print(f"     ⚠ 仲裁官输出解析失败，使用保守默认值")
        arb_data = {
            "upheld_severity":      original_sev,
            "severity_adjustment":  "维持",
            "adjustment_reason":    "仲裁输出解析异常，保留初始评级",
            "final_reasoning":      f"综合双方辩论，维持初始风险等级 {original_sev}",
            "key_points_upheld":    [],
            "key_points_rejected":  [],
            "consensus_whatToDo":   issue.get("whatToDo", []),
            "debate_consensus":     "部分争议",
            "arbitrator_confidence": 0.65,
        }

    # 确保必要字段存在
    arb_data.setdefault("upheld_severity",       original_sev)
    arb_data.setdefault("severity_adjustment",   "维持")
    arb_data.setdefault("adjustment_reason",     "")
    arb_data.setdefault("final_reasoning",       "")
    arb_data.setdefault("key_points_upheld",     [])
    arb_data.setdefault("key_points_rejected",   [])
    arb_data.setdefault("consensus_whatToDo",    issue.get("whatToDo", []))
    arb_data.setdefault("debate_consensus",      "部分争议")
    arb_data.setdefault("arbitrator_confidence", 0.7)

    # 归一化风险等级
    arb_data["upheld_severity"] = _normalize_severity(arb_data["upheld_severity"])

    # 计算辩论后置信度
    base_conf = float(arb_data.get("arbitrator_confidence", 0.7))
    # 分歧越大，置信度打折
    consensus_discount = {
        "全部裁定": 1.0,
        "部分争议": 0.9,
        "严重分歧": 0.75,
    }.get(arb_data["debate_consensus"], 0.85)
    final_confidence = round(min(1.0, base_conf * consensus_discount), 2)

    # ── 组装增强后的 issue ──────────────────────────────────
    debated_issue = issue.copy()
    debated_issue["severity"]          = arb_data["upheld_severity"]
    debated_issue["problem"]           = (
        f"【仲裁定论】{arb_data['final_reasoning']}\n\n"
        f"【原始分析】{problem_desc}"
        if arb_data.get("final_reasoning") else problem_desc
    )
    debated_issue["whatToDo"]          = arb_data["consensus_whatToDo"] or issue.get("whatToDo", [])
    debated_issue["confidence_score"]  = final_confidence
    debated_issue["model_agreement"]   = arb_data["debate_consensus"]
    debated_issue["models_flagged"]    = [
        roles["counsel_a"]["name"],
        roles["counsel_b"]["name"],
        roles["arbitrator"]["name"],
    ]

    # 调试元数据
    debated_issue["_debate_rounds"]          = debate_rounds
    debated_issue["_arbitrator_verdict"]     = {
        "model":               roles["arbitrator"]["name"],
        "upheld_severity":     arb_data["upheld_severity"],
        "severity_adjustment": arb_data["severity_adjustment"],
        "adjustment_reason":   arb_data.get("adjustment_reason", ""),
        "final_reasoning":     arb_data.get("final_reasoning", ""),
        "key_points_upheld":   arb_data.get("key_points_upheld", []),
        "key_points_rejected": arb_data.get("key_points_rejected", []),
        "consensus_whatToDo":  arb_data.get("consensus_whatToDo", []),
    }
    debated_issue["_original_severity"]      = original_sev
    debated_issue["_debated"]                = True

    sev_new = severity_order.get(arb_data["upheld_severity"], 2)
    sev_old = severity_order.get(original_sev, 2)
    print(
        f"     ✅ [{issue_title[:30]}] 仲裁完成：{original_sev} → {arb_data['upheld_severity']}"
        f"（{arb_data['severity_adjustment']}），置信度 {final_confidence}"
    )
    return debated_issue


def _run_debate_stage(
    issues: list[dict],
    contract_text: str,
    laws_context: str,
    category: str,
    lang_name: str = "Simplified Chinese (简体中文)",
    top_n: int = None,
    max_rounds: int = None,
    task_progress_cb=None,
    party_role: str = "partyA",
) -> tuple[list[dict], dict]:
    """
    ★ 辩论阶段主调度器（增强逻辑版）。
    策略：优先选择【风险等级高】且【置信度低（存在分歧）】的议题进行三方辩论。
    """
    top_n     = top_n     or DEBATE_TOP_N_ISSUES
    max_rounds = max_rounds or DEBATE_MAX_ROUNDS

    model_configs = _get_available_models()
    roles = _assign_debate_roles(model_configs)

    role_summary = {
        "counsel_a":  roles["counsel_a"]["name"],
        "counsel_b":  roles["counsel_b"]["name"],
        "arbitrator": roles["arbitrator"]["name"],
    }
    
    # ─── 核心修改点：定义辩论优先级权重 ───
    def get_debate_priority(iss):
        # 1. 风险等级分（基础分：40/30/20/10）
        severity_map = {"极高": 4, "高": 3, "中": 2, "低": 1}
        s_base = severity_map.get(_normalize_severity(iss.get("severity", "中")), 2) * 10
        
        # 2. 争议分（1.0 - 置信度）
        # 置信度越低（如0.4），争议分越高（0.6），排序越靠前
        conf = iss.get("confidence_score", 0.5)
        controversy_score = 1.0 - conf
        
        # 总分 = 风险权重 + 争议权重
        return s_base + controversy_score

    # 根据新逻辑排序，选出前 N 个议题
    # enumerate 是为了记住原始索引，方便后续替换
    indexed_issues = list(enumerate(issues))
    indexed_issues.sort(key=lambda x: get_debate_priority(x[1]), reverse=True)
    
    debate_indices = [idx for idx, iss in indexed_issues[:top_n]]
    skip_indices   = [idx for idx, iss in indexed_issues[top_n:]]

    print(f"\n  ⚖ 辩论庭组建完毕：甲方=[{roles['counsel_a']['name']}] 乙方=[{roles['counsel_b']['name']}] 仲裁=[{roles['arbitrator']['name']}]")
    print(f"  📋 选中辩论议题（按优先级）：{[issues[i].get('title','?')[:20] for i in debate_indices]}")
    print(f"  📋 跳过辩论议题：{len(skip_indices)} 个")

    updated_issues = [dict(iss) for iss in issues] # 深拷贝一份副本

    # 标记跳过的议题
    for idx in skip_indices:
        updated_issues[idx]["_debated"] = False

    if task_progress_cb:
        task_progress_cb(f"正在对 {len(debate_indices)} 个最具争议的高危条款展开三方辩论...")

    # ─── 以下执行逻辑保持不变 ───
    debate_results: dict[int, dict] = {}
    failed_indices: list[int] = []

    with ThreadPoolExecutor(max_workers=min(len(debate_indices), 3)) as ex:
        futures = {
            ex.submit(
                _debate_single_issue,
                issues[idx],
                contract_text,
                laws_context,
                roles,
                category,
                max_rounds,
                lang_name,
                party_role,
            ): idx
            for idx in debate_indices
        }
        try:
            for future in as_completed(futures, timeout=DEBATE_TIMEOUT):
                idx = futures[future]
                try:
                    debate_results[idx] = future.result()
                except Exception as e:
                    print(f"  ✗ 议题[{issues[idx].get('title','?')[:20]}]辩论失败: {e}")
                    failed_indices.append(idx)
        except FuturesTimeoutError:
            print(f"  ⚠ 辩论超时（>{DEBATE_TIMEOUT}s），整合已完成议题...")
            for f, idx in futures.items():
                if not f.done():
                    failed_indices.append(idx)

    # 合并结果
    for idx, debated in debate_results.items():
        updated_issues[idx] = debated
    for idx in failed_indices:
        updated_issues[idx]["_debated"]  = False
        updated_issues[idx]["_debate_failed"] = True

    # 最终结果排序：依然按严重度降序展示在前端
    severity_order = {"极高": 4, "高": 3, "中": 2, "低": 1}
    updated_issues.sort(
        key=lambda x: (
            severity_order.get(_normalize_severity(x.get("severity", "中")), 2),
            x.get("confidence_score", 0.5)
        ),
        reverse=True
    )

    # 重新生成 ID
    for i, iss in enumerate(updated_issues):
        iss["id"] = i + 1

    # 构造摘要
    debated_count  = len(debate_results)
    upgrade_count  = sum(1 for d in debate_results.values() if d.get("_arbitrator_verdict", {}).get("severity_adjustment") == "升级")
    downgrade_count = sum(1 for d in debate_results.values() if d.get("_arbitrator_verdict", {}).get("severity_adjustment") == "降级")
    
    debate_summary = {
        "enabled":           True,
        "debated_count":     debated_count,
        "skipped_count":     len(skip_indices),
        "failed_count":      len(failed_indices),
        "upgrades":          upgrade_count,
        "downgrades":        downgrade_count,
        "maintained":        debated_count - upgrade_count - downgrade_count,
        "rounds_per_issue":  max_rounds,
        "roles":             role_summary,
    }
    return updated_issues, debate_summary

KB_MANIFEST_PATH = './chroma_db/kb_manifest.json'

def _load_kb_manifest() -> dict:
    """加载知识库版本清单"""
    try:
        if os.path.exists(KB_MANIFEST_PATH):
            with open(KB_MANIFEST_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def get_law_docs_enhanced(
    contract_text: str,
    category: str,
    k: int = 6,
    include_cases: bool = True,
) -> tuple[list, int, dict]:
    """
    ★ 增强版法律文档检索，感知文档类型。

    检索策略：
    1. 分类专属库：MMR 检索（k 条）
    2. 通用库：补充基础法（3 条）
    3. 案例库（可选）：专项检索典型案例（2 条），用于类案比较分析

    返回 (all_docs, total_count, doc_type_summary)
    doc_type_summary: {"法律法规": N, "司法解释": N, "典型案例": N, "部门规章": N}
    """
    all_docs: list = []
    seen_contents: set = set()
    doc_type_summary: dict = {}

    def _mmr_search(persist_dir: str, query: str, k_val: int,
                    doc_type_filter: str = None) -> list:
        if not os.path.exists(persist_dir):
            return []
        try:
            db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            # 如果需要按文档类型过滤
            where_filter = {"document_type": doc_type_filter} if doc_type_filter else None
            if where_filter:
                results = db.max_marginal_relevance_search(
                    query, k=k_val, fetch_k=k_val * 3, lambda_mult=0.7,
                    filter=where_filter
                )
            else:
                results = db.max_marginal_relevance_search(
                    query, k=k_val, fetch_k=k_val * 3, lambda_mult=0.7
                )
            return results
        except Exception as e:
            print(f"  ⚠ MMR 检索失败（{persist_dir}）: {e}，回退到相似度检索...")
            try:
                db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
                return db.similarity_search(query, k=k_val)
            except Exception as e2:
                print(f"  ✗ 向量检索彻底失败: {e2}")
                return []

    def _add_docs(docs: list):
        for doc in docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
                # 统计文档类型
                dtype = doc.metadata.get('document_type', '法律法规')
                doc_type_summary[dtype] = doc_type_summary.get(dtype, 0) + 1

    # 1. 分类专属库（法律法规 + 司法解释）
    cat_path = os.path.join("./chroma_db", category)
    _add_docs(_mmr_search(cat_path, contract_text, k))

    # 2. 通用库（基础法）
    general_path = os.path.join("./chroma_db", "通用")
    if general_path != cat_path:
        _add_docs(_mmr_search(general_path, contract_text, 3))

    # 3. 典型案例库（可选，用于类案类比）
    if include_cases:
        cases_path = os.path.join("./chroma_db", "典型案例")
        case_docs = _mmr_search(cases_path, contract_text, 2)
        if case_docs:
            _add_docs(case_docs)
            print(f"  ⚖ 检索到 {len(case_docs)} 条典型案例")

    if not all_docs:
        print(f"  ⚠ 分类库 [{category}] 和通用库均未检索到法条，LLM 将依赖训练知识作答")

    return all_docs, len(all_docs), doc_type_summary


def _format_law_context(docs: list) -> str:
    """将检索到的法律文档格式化为 Prompt 上下文，区分文档类型"""
    if not docs:
        return "（未检索到匹配法条，请依据专业知识作答）"

    parts = []
    doc_type_icons = {
        '法律法规': '📜',
        '司法解释': '⚖️',
        '典型案例': '🏛️',
        '部门规章': '📋',
    }
    for i, doc in enumerate(docs):
        dtype = doc.metadata.get('document_type', '法律法规')
        law_name = doc.metadata.get('law_name', '未知法律')
        icon = doc_type_icons.get(dtype, '📄')
        parts.append(
            f"【参考法条{i+1}】{icon}[{dtype}·{law_name}]：\n{doc.page_content}"
        )

    return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════════
#  原有工具函数（保持不变）
# ════════════════════════════════════════════════════════════════

def robust_json_cleaner(text):
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            content = text[start:end + 1]
            content = content.replace('```json', '').replace('```', '').strip()
            return content
        return text
    except Exception as e:
        print(f"JSON清理失败: {e}")
        return text


def make_llm(max_tokens=3000):
    """创建主力 LLM 实例（DeepSeek）"""
    return ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com",
        max_tokens=max_tokens,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )


# ════════════════════════════════════════════════════════════════
#  ★ 核心分析引擎 v3.0（多模型 + 增强知识库 + 律师审查标记）
# ════════════════════════════════════════════════════════════════

def run_deep_analysis(task_id, contract_text, category, language='zh-CN', audit_id: str = None, party_role: str = 'partyA'):
    """
    合同深度分析（v4.0）：
    Stage 1   ── 多模型并行风险审查 + 共识合并（★增强A 快思考）
    Stage 1.5 ── 多智能体辩论仲裁（★增强A v4.0 慢思考，ENABLE_DEBATE_MODE=true 时激活）
                  甲方律师 vs 乙方律师 → 仲裁官裁定 → 风险等级二次校准
    Stage 2a  ── 谈判邮件生成（基于辩论后风险点）
    Stage 2b  ── 多风格谈判话术
    Stage 3   ── 完整修订合同
    Stage 4   ── 案例类比分析（★增强B，有案例时）
    Stage 5   ── 整合 + 律师审查建议（★增强C）
    party_role: 'partyA'（甲方）或 'partyB'（乙方），决定审查视角与谈判立场
    """
    # ── 甲方/乙方角色描述 ──────────────────────────────────────────
    party_label   = "甲方" if party_role == 'partyA' else "乙方"
    party_opposite = "乙方" if party_role == 'partyA' else "甲方"
    party_desc    = (
        "合同的发起方、提供方或服务/商品提供者"
        if party_role == 'partyA'
        else "合同的接受方、签署方或服务/商品接收者"
    )
    party_risk_focus = (
        f"重点识别：对{party_label}的不公平责任条款、权利限制条款、"
        f"可能被{party_opposite}利用的漏洞、{party_label}的潜在违约风险，"
        f"以及{party_opposite}单方面获益的隐性条款。"
        if party_role == 'partyA'
        else
        f"重点识别：{party_opposite}强加给{party_label}的苛刻义务、"
        f"对{party_label}权益的限制或剥夺、{party_label}面临的不对等风险、"
        f"违约金/索赔上限不合理条款，以及{party_label}的合法权益保障缺失。"
    )
    text_hash = _contract_hash(contract_text, category, language, party_role)
    models_used = [make_llm.__name__]  # 默认

    try:
        lang_name = LANGUAGE_MAP.get(language, 'Simplified Chinese (简体中文)')
        lang_instruction = (
            f"CRITICAL LANGUAGE REQUIREMENT: You MUST write ALL output text "
            f"(titles, summaries, explanations, JSON string values, everything) "
            f"in {lang_name}. Do NOT use any other language for the output values."
        )

        # ── 缓存命中检查 ──
        cached = _get_cached_result(text_hash, category, language)
        if cached:
            print(f"任务 {task_id}: ⚡ 缓存命中（hit_count={cached.get('_cache_hits')}）")
            tasks[task_id] = {
                "status":   "completed",
                "result":   cached,
                "progress": "分析完成！（缓存极速返回）",
                "_completed_at": time.time(),
            }
            _log_analysis_audit(audit_id, '', text_hash, category, language,
                                 status='completed', cached=True, ended=True)
            return

        # ★ 增强 B：加载知识库版本清单
        kb_manifest = _load_kb_manifest()

        # ────────── Stage 1: 多模型并行风险审查 ──────────
        tasks[task_id]["progress"] = "正在组建多模型审查团，查阅权威法条..."
        tasks[task_id]["stage"] = 1

        docs, law_doc_count, doc_type_summary = get_law_docs_enhanced(
            contract_text, category, k=6, include_cases=True
        )
        laws_context = _format_law_context(docs)

        # 分离案例文档供后续专项分析
        case_docs = [d for d in docs if d.metadata.get('document_type') == '典型案例']

        if category == '网络数字类':
            prompt_1 = f"""{lang_instruction}

你是一位专注于互联网法律合规的资深法律专家，深度熟悉《个人信息保护法》《网络安全法》《数据安全法》《消费者权益保护法》《App违法违规收集使用个人信息行为认定方法》等核心法规。
你正在为用户（本协议的【{party_label}】/普通用户）对一份网络平台用户协议/服务条款/隐私政策进行专项风险审查。

任务：深度审查【待审协议】，站在【用户（{party_label}）】的立场，识别协议中损害用户权益、超范围授权或违反法律规定的风险条款。
重点识别方向：①过宽或隐晦的数据收集与使用条款 ②不透明的第三方信息共享 ③单方面免责或转移责任的条款 ④单方面变更协议或终止服务条款 ⑤对用户知情权、删除权、撤回同意权的限制 ⑥内容版权过度转让条款 ⑦不公平的争议解决条款

【参考法律依据】：
{laws_context}

【待审协议内容】：
{contract_text}

【要求】：
1. 从普通用户视角识别风险，plainLanguage 必须用通俗易懂的语言（不用法律术语）解释。
2. 每个风险点必须提供具体法律依据（法律名称+条文编号+条文核心内容）。
3. 给出 riskScore (0-100)，反映该协议对用户权益的整体威胁程度。
4. 数量：识别最核心的 5-8 个风险点。
5. clauseText 必须是协议原文的完整、逐字摘录。
6. whatToDo 应给出普通用户实际可操作的建议（如：可要求删除数据、可撤回授权等）。

请输出 JSON（所有字符串值使用 {lang_name}）：
{{
  "contractType": "协议类型（如：APP用户服务协议+隐私政策）",
  "jurisdiction": "适用地区",
  "overallRisk": "极高/高/中/低",
  "riskScore": 整数,
  "summary": "一句通俗的总体点评（从普通用户角度说明这份协议的整体风险）",
  "issues": [
    {{
      "id": 1,
      "severity": "极高/高/中/低",
      "title": "风险标题（通俗表达，如：平台可随意分享你的个人信息给第三方）",
      "clauseText": "逐字摘录协议原文（相关条款）",
      "lawReference": "具体法律名称+条文编号+条文核心内容",
      "plainLanguage": ["通俗解释：用普通用户能理解的语言说明这条款的实际含义和危害"],
      "problem": "专业风险分析：该条款违反了什么规定，对用户有什么具体危害",
      "whatToDo": ["可操作建议：用户遇到此情况可以怎么做（具体、可执行）"],
      "alternative": "建议的合规修改方向（保护用户权益的改进措辞）"
    }}
  ]
}}"""
        else:
            prompt_1 = f"""{lang_instruction}

你是一位拥有顶级事务所背景的资深法律合伙人，擅长从细节中洞察法律风险。
你正在为合同的【{party_label}】（{party_desc}）提供专属法律审查服务。

任务：深度审查【待审合同】，站在【{party_label}】的立场，结合【{category}】领域的【法律依据】进行穿透式分析。
{party_risk_focus}

【参考法律依据】：
{laws_context}

【待审合同内容】：
{contract_text}

【要求】：
1. 从【{party_label}】的视角识别隐藏陷阱、责任不对等、关键条款缺失等，必须极其专业深刻。
2. 每个风险点必须提供 lawReference（具体法律名称、条目、内容）。
3. 给出 riskScore (0-100)，反映该合同对【{party_label}】的整体风险水平。
4. 数量：请仅识别【最核心、风险等级最高】的 5-7 个风险点，严禁超过 8 个。
5. clauseText 必须是合同原文中的完整、逐字摘录，不要改动任何字符。
6. 请审慎评估风险，避免过度夸大风险等级。

请输出 JSON（所有字符串值使用 {lang_name}）：
{{
  "contractType": "合同类型",
  "jurisdiction": "管辖地",
  "overallRisk": "极高/高/中/低 (translated)",
  "riskScore": 整数,
  "summary": "一句客观全面的点评（明确指出对{party_label}的整体风险评估）",
  "issues": [
    {{
      "id": 1,
      "severity": "极高/高/中/低 (translated)",
      "title": "风险标题",
      "clauseText": "逐字摘录合同原文",
      "lawReference": "具体法律名称、条目、内容",
      "plainLanguage": ["大白话解释（从{party_label}角度）"],
      "problem": "深度风险剖析（分析该条款对{party_label}的具体危害）",
      "whatToDo": ["精准行动对策（站在{party_label}立场的建议）"],
      "alternative": "防御性修订建议（保护{party_label}权益的具体条款替换文字）"
    }}
  ]
}}"""

        # ★ 多模型并行审查
        data_1, models_used = _multi_model_risk_review(prompt_1)
        print(f"任务 {task_id}: Stage 1 完成，使用模型：{models_used}")

        # ★ 流式推送：Stage 1 完成后立即把识别到的风险点存入任务字典
        # /analyze/stream 的 watcher 线程会即时取走并推送到前端
        tasks[task_id]["_stage1_issues"]  = data_1.get("issues", [])
        tasks[task_id]["_stage1_meta"] = {
            "contractType": data_1.get("contractType", ""),
            "overallRisk":  data_1.get("overallRisk", ""),
            "riskScore":    data_1.get("riskScore", 0),
            "summary":      data_1.get("summary", ""),
            "jurisdiction": data_1.get("jurisdiction", ""),
        }
        tasks[task_id]["_stage1_ready"] = True

        # ────────── Stage 1.5: 多智能体辩论仲裁（★增强A v4.0）──────────
        debate_summary = {"enabled": False}
        if ENABLE_DEBATE_MODE and data_1.get('issues'):
            tasks[task_id]["progress"] = (
                f"⚖ 正在对 {min(DEBATE_TOP_N_ISSUES, len(data_1['issues']))} 个高危条款"
                f"展开三方辩论仲裁（甲方律师 vs 乙方律师 → 仲裁官裁定）..."
            )
            tasks[task_id]["stage"] = 1  # 仍在 Stage 1 阶段

            def _progress_cb(msg):
                tasks[task_id]["progress"] = msg

            try:
                debated_issues, debate_summary = _run_debate_stage(
                    issues        = data_1['issues'],
                    contract_text = contract_text,
                    laws_context  = laws_context,
                    category      = category,
                    lang_name     = lang_name,
                    top_n         = DEBATE_TOP_N_ISSUES,
                    max_rounds    = DEBATE_MAX_ROUNDS,
                    task_progress_cb = _progress_cb,
                    party_role    = party_role,
                )
                data_1['issues'] = debated_issues
                print(f"任务 {task_id}: Stage 1.5 辩论仲裁完成 → "
                      f"升级 {debate_summary['upgrades']} 个，"
                      f"降级 {debate_summary['downgrades']} 个")

                # 辩论后重新加权风险总分（取仲裁后最高危3个issue的均值）
                sev_map = {"极高": 80, "高": 60, "中": 40, "低": 20}
                top_scores = sorted(
                    [sev_map.get(i.get("severity","中"), 45) for i in debated_issues],
                    reverse=True
                )[:3]
                if top_scores:
                    new_score = int(sum(top_scores) / len(top_scores))
                    # 温和调整，不完全覆盖并行阶段分数
                    data_1['riskScore'] = int(data_1.get('riskScore', 50) * 0.4 + new_score * 0.6)

            except Exception as e:
                print(f"  ⚠ Stage 1.5 辩论失败（不影响主流程）: {e}")
                debate_summary = {"enabled": False, "error": str(e)}

        llm = make_llm(max_tokens=3000)

        # ────────── Stage 2a: 长邮件生成 ──────────
        tasks[task_id]["progress"] = "正在生成详尽谈判邮件..."
        tasks[task_id]["stage"] = 2
        issues_brief = json.dumps(data_1['issues'], ensure_ascii=False)

        prompt_2a = f"""{lang_instruction}

你是一位资深商务律师和谈判专家。你的委托人是本合同的【{party_label}】（{party_desc}）。
基于以下法律风险点，以【{party_label}】名义起草一封致对方（{party_opposite}）的商务谈判长邮件，
要求对方就以下风险条款作出修改或补充承诺。
【风险点摘要】：{issues_brief}

【要求】：
1. 邮件内容 ("email")：必须极其详尽，对每个关键风险点进行专业化阐述，立场清晰站在{party_label}一方。
2. 格式：严格遵守商务邮件格式，分段清晰，使用专业辞令。
3. 字数：不少于 500 字，展现极高的专业度和诚意。
4. 严禁使用双引号，引用请用单引号。
5. talkTrack 话术要自然，不要假定对方姓氏。
6. strategy 应体现作为【{party_label}】的整体谈判方针。

请输出 JSON：
{{
  "strategy": "作为{party_label}的总体博弈方针简述",
  "email": "500字以上的详尽谈判邮件全文（以{party_label}名义致{party_opposite}）..."
}}"""

        res_2a = _retry_llm(lambda: llm.invoke(prompt_2a))
        data_2a = json.loads(robust_json_cleaner(res_2a.content), strict=False)

        # ────────── Stage 2b: 多风格谈判话术 ──────────
        tasks[task_id]["progress"] = "正在生成多方案谈判话术..."
        tasks[task_id]["stage"] = 2

        prompt_2b = f"""{lang_instruction}

基于以下法律风险点，以本合同【{party_label}】（{party_desc}）的立场，
设计多维度的口头谈判脚本和不同风格的应对方案，用于与{party_opposite}进行谈判。
【风险点摘要】：{issues_brief}

【话术要求】：
1. talkTrack：包含自然的开场白和 3 个核心说服理由，立场鲜明代表{party_label}。
2. styles：提供强硬、协商、妥协三种截然不同的完整博弈逻辑，均以{party_label}利益最大化为目标。
3. 话术要自然，不要假定对方姓氏与性别。

请输出 JSON：
{{
  "talkTrack": {{
    "opening": "{party_label}开场白话术...",
    "reasons": ["{party_label}核心理由1", "理由2", "理由3"]
  }},
  "styles": {{
    "aggressive": "强硬风格：{party_label}坚守底线、对{party_opposite}施压的具体论点和话术...",
    "consultative": "协商风格：{party_label}寻求共赢、引导{party_opposite}接受修改的话术与方案...",
    "compromise": "妥协风格：{party_label}底线保障与可让步的折中条件..."
  }}
}}"""

        res_2b = _retry_llm(lambda: llm.invoke(prompt_2b))
        data_2b = json.loads(robust_json_cleaner(res_2b.content), strict=False)

        # ────────── Stage 3: 完整修订合同 ──────────
        tasks[task_id]["progress"] = "正在生成完整修订版合同..."
        tasks[task_id]["stage"] = 3

        llm_large = make_llm(max_tokens=4000)

        prompt_3 = f"""{lang_instruction}

你是一位精通合同起草的资深法律顾问。你的委托人是本合同的【{party_label}】（{party_desc}）。
请基于【原始合同】和【已识别的风险点及修订建议】，生成一份完整的修订版合同，
修订方向应以保护和维护【{party_label}】的合法权益为核心原则，同时确保合同内容合法合规。

【原始合同】：
{contract_text}

【风险点及修订建议】：
{issues_brief}

【任务要求】：
1. 保留原合同的完整结构、条款编号和所有未涉及风险的条款原文。
2. 对每个风险条款，应用其对应的 "alternative"（修订建议）进行替换或补充，优先保护【{party_label}】权益。
3. 如有缺失的重要条款（尤其是保护【{party_label}】的条款），在合适位置补充完整。
4. 修订处用 【修订】 标记开头，便于对照查看。
5. revisionNotes 列出每处修订的简要说明，并注明该修订如何保护了【{party_label}】。

请输出 JSON：
{{
  "revisedContract": "完整修订合同全文（修订处以【修订】标记）...",
  "revisionNotes": [
    {{"clauseRef": "条款编号或名称", "change": "修订说明（含对{party_label}的保护作用）"}}
  ],
  "revisionSummary": "本次修订的整体说明，重点阐述对【{party_label}】的保护效果（100字以内）"
}}"""

        res_3 = _retry_llm(lambda: llm_large.invoke(prompt_3))
        data_3 = json.loads(robust_json_cleaner(res_3.content), strict=False)

        # ────────── Stage 3.5: 用户协议权益专项分析（仅限网络数字类）──────────
        data_tos_analysis = {}
        if category == '网络数字类':
            tasks[task_id]["progress"] = "正在分析用户协议权益授权、用户权益与缺失权益..."
            tasks[task_id]["stage"] = 3
            print(f"任务 {task_id}: 执行 Stage 3.5（网络数字类用户协议专项分析）...")

            prompt_tos = f"""{lang_instruction}

你是一位专注于互联网法律合规和用户权益保护的资深法律专家，深度熟悉以下核心法规：
《个人信息保护法》（PIPL）、《网络安全法》、《数据安全法》、《消费者权益保护法》、
《电信和互联网用户个人信息保护规定》、《互联网信息服务管理办法》、
《App违法违规收集使用个人信息行为认定方法》、《移动互联网应用程序个人信息保护管理暂行规定》等。

你正在审查的是一份【网络平台用户协议/服务条款/隐私政策】。请对该协议进行三个维度的全面权益分析。

【参考法律依据】：
{laws_context}

【待审用户协议原文】：
{contract_text}

═══════════════════════════════════════════════════════
【分析维度一：权益授权分析】
用户在签署本协议时，同意授权给平台哪些权益和数据？

注意：
- 穿透分析隐晦措辞（如"改善用户体验"可能隐含行为追踪）
- 区分"必要授权"（服务正常运行所需）与"扩展授权"（超出服务核心范围）
- 重点关注：个人数据收集、设备权限、第三方共享、商业使用、内容权利转让等
═══════════════════════════════════════════════════════
【分析维度二：用户权益分析】
用户在法律保护下享有哪些权益？平台对应的法定义务是什么？

注意：
- 结合《个人信息保护法》第44-47条、《消费者权益保护法》等核心权利条款
- 明确标注协议是否已保障该权益（isGuaranteed: 协议中有明确承诺则为true）
- 每项权益提供2-3个通俗易懂的生活场景举例
═══════════════════════════════════════════════════════
【分析维度三：权益缺失分析】
对照法律法规要求，本协议在用户权益保护方面存在哪些缺失或不足？

注意：
- 必须引用具体法律名称、条文编号和条文内容（不得泛泛而谈）
- 区分严重缺失（违法风险）与一般不足（不规范但未必违法）
- 用通俗语言解释缺失对普通用户的实际影响
═══════════════════════════════════════════════════════

请输出 JSON（使用 {lang_name}）：
{{
  "authorizedRights": [
    {{
      "category": "授权类型（从以下选择：个人信息类/行为数据类/设备权限类/内容权利类/商业使用类/第三方共享类/其他）",
      "item": "授权项目名称（简洁，10字以内）",
      "detail": "协议中的具体授权内容描述",
      "plainExplanation": "通俗解释：用普通人能理解的语言说明这究竟意味着什么，实际影响是什么",
      "isNecessary": true/false（是否为服务正常运行所必须的授权）,
      "riskLevel": "高/中/低",
      "clauseHint": "涉及的条款关键词或位置提示（10字以内）"
    }}
  ],
  "userRights": [
    {{
      "rightName": "权益名称（简洁）",
      "description": "该权益的完整说明",
      "plainExplanation": "通俗解释：用一两句话让普通用户明白这个权益是什么",
      "platformObligation": "平台对应的法定义务",
      "isGuaranteed": true/false（本协议是否明确承诺保障此权益）,
      "guaranteeDetail": "若已保障，协议中的相关承诺（未保障则填'协议未明确'）",
      "scenarios": [
        {{
          "scene": "具体生活场景描述（让普通用户有代入感）",
          "yourRight": "在这个场景中，您有权做什么",
          "platformShouldDo": "平台在这个场景中应当怎么做"
        }}
      ]
    }}
  ],
  "missingRights": [
    {{
      "missingItem": "缺失的权益保护项目",
      "severity": "严重/中等/轻微",
      "legalBasis": "具体法律依据（必须包含：法律名称 + 具体条文编号 + 条文核心内容）",
      "currentStatus": "当前协议状态（缺失/表述模糊/不完整/存在矛盾）",
      "plainImpact": "通俗影响说明：用普通用户能理解的语言，说明这个缺失对用户有什么实际危害",
      "suggestion": "建议的改进方向（具体可操作）"
    }}
  ]
}}"""

            try:
                res_tos = _retry_llm(lambda: llm.invoke(prompt_tos))
                data_tos_analysis = json.loads(robust_json_cleaner(res_tos.content), strict=False)
                print(f"任务 {task_id}: Stage 3.5 用户协议权益专项分析完成 "
                      f"（授权项 {len(data_tos_analysis.get('authorizedRights', []))} 条，"
                      f"用户权益 {len(data_tos_analysis.get('userRights', []))} 条，"
                      f"缺失权益 {len(data_tos_analysis.get('missingRights', []))} 条）")
            except Exception as e:
                print(f"  ⚠ Stage 3.5 用户协议分析失败（不影响主流程）: {e}")
                data_tos_analysis = {}

        # ────────── Stage 4: 典型案例类比分析（★增强B）──────────
        data_case_analysis = {}
        if case_docs:
            tasks[task_id]["progress"] = "正在进行典型案例类比分析..."
            tasks[task_id]["stage"] = 4
            print(f"任务 {task_id}: 执行 Stage 4（案例类比分析，{len(case_docs)} 条案例）...")

            cases_text = "\n\n".join([
                f"【案例{i+1}·{d.metadata.get('law_name','未知')}】：\n{d.page_content}"
                for i, d in enumerate(case_docs)
            ])

            prompt_4 = f"""{lang_instruction}

你是一位熟悉司法实践的法律专家。请将【待审合同风险点】与【典型案例】进行类比分析，
评估本合同可能的司法结果。分析视角以本合同【{party_label}】（{party_desc}）的立场为主。

【待审合同类型】：{data_1.get('contractType', '未知')}
【主要风险点】：{json.dumps(data_1['issues'][:3], ensure_ascii=False)}

【典型参考案例】：
{cases_text}

【分析要求】：
- 指出哪些风险与典型案例高度相似
- 预测若发生纠纷，作为{party_label}可能面临的裁判结果
- 给出基于案例的、针对{party_label}的专项防范建议

请输出 JSON：
{{
  "caseComparison": [
    {{
      "issueTitle": "风险标题",
      "similarCase": "相似案例名称",
      "similarity": "高/中/低",
      "predictedOutcome": "若{party_label}诉诸法律的可能裁判结果",
      "caseBasedAdvice": "基于案例的、针对{party_label}的防范建议"
    }}
  ],
  "overallCaseInsight": "从{party_label}角度的整体司法风险评估（100字以内）"
}}"""

            try:
                res_4 = _retry_llm(lambda: llm.invoke(prompt_4))
                data_case_analysis = json.loads(robust_json_cleaner(res_4.content), strict=False)
                print(f"任务 {task_id}: Stage 4 案例类比分析完成")
            except Exception as e:
                print(f"  ⚠ Stage 4 案例类比失败（不影响主流程）: {e}")

        # ────────── Stage 5: 整合 + 律师审查建议（★增强C）──────────
        tasks[task_id]["progress"] = "正在整合分析报告..."
        tasks[task_id]["stage"] = 5

        final_result = data_1
        final_result['negotiation'] = {
            "strategy":  data_2a.get('strategy', ''),
            "email":     data_2a.get('email', ''),
            "talkTrack": data_2b.get('talkTrack', {}),
            "styles":    data_2b.get('styles', {}),
        }
        final_result['revisedContract']  = data_3.get('revisedContract', '')
        final_result['revisionNotes']    = data_3.get('revisionNotes', [])
        final_result['revisionSummary']  = data_3.get('revisionSummary', '')

        # ★ 网络数字类专项：用户协议权益分析
        if data_tos_analysis:
            final_result['tosAnalysis'] = data_tos_analysis

        # ★ 增强B：案例分析 + 知识库元数据
        if data_case_analysis:
            final_result['caseAnalysis'] = data_case_analysis
        final_result['_doc_type_summary']  = doc_type_summary
        final_result['_kb_updated_at']     = kb_manifest.get('last_updated', '未知')
        final_result['_kb_law_count']      = kb_manifest.get('total_laws', 0)

        # ★ 增强A：多模型元数据 + 辩论仲裁元数据
        final_result['_law_doc_count']     = law_doc_count
        final_result['_analysis_version']  = ANALYSIS_VERSION
        final_result['_cached']            = False
        final_result['_models_used']       = models_used
        final_result['_multi_model']       = data_1.get('_multi_model', False)
        final_result['_overall_confidence'] = data_1.get('_overall_confidence', None)
        # ★ v4.0 辩论仲裁元数据
        final_result['_debate_enabled']    = debate_summary.get('enabled', False)
        final_result['_debate_summary']    = debate_summary
        # ★ 甲方/乙方角色元数据
        final_result['_party_role']        = party_role
        final_result['_party_label']       = party_label

        # ★ 增强C：高风险自动建议律师审查
        risk_score = final_result.get('riskScore', 0)
        lawyer_review_suggested = risk_score >= AUTO_SUGGEST_REVIEW_SCORE
        final_result['_lawyer_review_suggested'] = lawyer_review_suggested
        if lawyer_review_suggested:
            final_result['_lawyer_review_reason'] = (
                f"合同风险分达到 {risk_score}（阈值 {AUTO_SUGGEST_REVIEW_SCORE}），"
                f"建议申请专业律师复核，进一步降低法律风险。"
            )

        tasks[task_id] = {
            "status":   "completed",
            "result":   final_result,
            "progress": "分析完成！",
            "_completed_at": time.time(),
        }
        print(
            f"任务 {task_id}: 全流程分析已完成"
            f"（法条引用 {law_doc_count} 条，模型: {models_used}，"
            f"律师审查建议: {'是' if lawyer_review_suggested else '否'}）"
        )

        _save_to_cache(text_hash, category, language, final_result)
        _log_analysis_audit(audit_id, '', text_hash, category, language,
                             status='completed', ended=True, models_used=models_used)

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"任务 {task_id} 异常详情: {error_detail}")
        tasks[task_id] = {
            "status":      "failed",
            "error":       str(e),
            "error_detail": error_detail,
            "progress":    "分析失败",
            "_completed_at": time.time(),
        }
        _log_analysis_audit(audit_id, '', text_hash, category, language,
                             status='failed', ended=True)


# ════════════════════════════════════════════════════════════════
#  Flask 应用初始化
# ════════════════════════════════════════════════════════════════

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()


init_db()
tasks = {}

LANGUAGE_MAP = {
    'zh-CN': 'Simplified Chinese (简体中文)',
    'zh-TW': 'Traditional Chinese (繁體中文)',
    'en':    'English',
    'ja':    'Japanese (日本語)',
    'ko':    'Korean (한국어)',
    'fr':    'French (Français)',
    'de':    'German (Deutsch)',
    'es':    'Spanish (Español)',
    'pt':    'Portuguese (Português)',
    'ar':    'Arabic (العربية)',
    'ru':    'Russian (Русский)',
}

@app.route('/debug-files')
def debug_files():
    import os
    res = {
        "current_work_dir": os.getcwd(),
        "files_in_root": os.listdir('.'),
    }
    # 检查 static 文件夹
    if os.path.exists('static'):
        res["static_content"] = os.listdir('static')
        if os.path.exists('static/img'):
            res["img_content"] = os.listdir('static/img')
        else:
            res["img_content"] = "FOLDER 'img' NOT FOUND"
    else:
        res["static_content"] = "FOLDER 'static' NOT FOUND"
    return jsonify(res)

# 兼容旧版 get_law_docs 引用
def get_law_docs(contract_text: str, category: str, k: int = 6) -> tuple[list, int]:
    docs, count, _ = get_law_docs_enhanced(contract_text, category, k=k, include_cases=False)
    return docs, count


# ════════════════════════════════════════════════════════════════
#  核心分析路由（原有 /analyze、/status、/analyze/quick 保持不变）
# ════════════════════════════════════════════════════════════════

@app.route('/analyze', methods=['POST'])
def analyze():
    """启动合同分析任务（异步），v3.0 新增多模型审查"""
    try:
        ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
        allowed, remaining = _check_analysis_rate_limit(ip)
        if not allowed:
            return jsonify({
                "error": "rate_limited",
                "message": f"请求过于频繁，每小时最多 {ANALYSIS_RATE_LIMIT_PER_HOUR} 次分析"
            }), 429

        data = request.json or {}
        raw_text = data.get('text', '').strip()
        category = data.get('category', '其他类')
        language = data.get('language', 'zh-CN')
        party_role = data.get('party_role', 'partyA')  # 'partyA' 或 'partyB'
        if party_role not in ('partyA', 'partyB'):
            party_role = 'partyA'
        if language not in LANGUAGE_MAP:
            language = 'zh-CN'

        if not raw_text:
            return jsonify({"error": "无合同内容"}), 400

        try:
            contract_text, warnings = _sanitize_contract_text(raw_text)
        except ValueError as e:
            return jsonify({"error": "invalid_input", "message": str(e)}), 400

        task_id  = str(uuid.uuid4())
        audit_id = str(uuid.uuid4())
        text_hash = _contract_hash(contract_text, category, language, party_role)

        tasks[task_id] = {
            "status":   "processing",
            "stage":    0,
            "progress": f"正在初始化【{category}】多模型审查任务...",
            "_created_at": time.time(),
        }

        _log_analysis_audit(audit_id, ip, text_hash, category, language)

        threading.Thread(
            target=run_deep_analysis,
            args=(task_id, contract_text, category, language, audit_id, party_role),
            daemon=True
        ).start()

        return jsonify({
            "task_id":  task_id,
            "warnings": warnings,
            "rate_limit_remaining": remaining,
            "multi_model_enabled":  ENABLE_MULTI_MODEL,
            "party_role": party_role,
        })
    except Exception as e:
        return jsonify({"error": f"请求处理失败: {str(e)}"}), 500


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found"}), 404
    if "progress" not in task:
        task["progress"] = "处理中..."
    return jsonify(task)


@app.route('/analyze/quick', methods=['POST'])
def analyze_quick():
    """快速预扫描（同步，约 10-15 秒）"""
    try:
        data = request.json or {}
        raw_text = data.get('text', '').strip()
        category = data.get('category', '其他类')
        language = data.get('language', 'zh-CN')
        party_role = data.get('party_role', 'partyA')
        if party_role not in ('partyA', 'partyB'):
            party_role = 'partyA'
        party_label   = "甲方" if party_role == 'partyA' else "乙方"
        party_opposite = "乙方" if party_role == 'partyA' else "甲方"

        if not raw_text:
            return jsonify({"error": "无合同内容"}), 400

        try:
            contract_text, warnings = _sanitize_contract_text(raw_text)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        lang_name = LANGUAGE_MAP.get(language, 'Simplified Chinese (简体中文)')
        llm = make_llm(max_tokens=1500)
        docs, doc_count, _ = get_law_docs_enhanced(contract_text, category, k=3, include_cases=False)
        laws_brief = "\n".join([
            f"[{d.metadata.get('law_name','?')}]: {d.page_content[:200]}"
            for d in docs[:3]
        ])

        prompt = f"""你是法律专家。以本合同【{party_label}】的视角，快速扫描以下合同，识别对【{party_label}】最危险的 3 个风险点。
【合同分类】：{category}
【审查视角】：{party_label}（对方为{party_opposite}）
【参考法条摘要】：{laws_brief or '（无检索结果）'}
【合同内容（前3000字）】：{contract_text[:3000]}

输出 JSON（使用 {lang_name}）：
{{
  "quickRiskLevel": "极高/高/中/低",
  "quickScore": 整数0-100,
  "contractType": "推断的合同类型",
  "topThreats": [
    {{"title": "风险标题", "severity": "极高/高/中/低", "brief": "一句话说明（对{party_label}的危害）"}}
  ],
  "quickTip": "对{party_label}最重要的一条建议（20字以内）",
  "lawyerReviewSuggested": true/false
}}"""

        res = _retry_llm(lambda: llm.invoke(prompt))
        result = json.loads(robust_json_cleaner(res.content), strict=False)
        result['_law_doc_count'] = doc_count
        result['_party_role']   = party_role
        result['_party_label']  = party_label
        result['warnings'] = warnings
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"快速分析失败: {str(e)}"}), 500


@app.route('/ocr-refine', methods=['POST'])
def ocr_refine():
    try:
        data = request.json or {}
        raw_text = data.get('text', '').strip()
        language = data.get('language', 'zh-CN')
        lang_name = LANGUAGE_MAP.get(language, 'Simplified Chinese (简体中文)')

        if not raw_text:
            return jsonify({"error": "无文本内容"}), 400

        llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com",
            max_tokens=4000,
            temperature=0.1,
        )

        prompt = f"""You are a professional document OCR correction expert.
The following text was extracted via OCR and may contain formatting issues.
Please clean and reconstruct this text into properly formatted, readable contract text.
Preserve ALL original content and meaning — do NOT add, remove or alter any substantive terms.
Output the refined text as plain text only (no JSON, no markdown).

Raw OCR text:
{raw_text}"""

        res = llm.invoke(prompt)
        return jsonify({"refined": res.content.strip()})

    except Exception as e:
        return jsonify({"error": f"OCR优化失败: {str(e)}"}), 500


@app.route('/languages', methods=['GET'])
def get_languages():
    return jsonify(LANGUAGE_MAP)


@app.route('/deploy/info', methods=['GET'])
def get_deploy_info():
    return jsonify({
        "deployment_mode": "onprem",
        "encryption_enabled": True,
        "crypto_library": True,
        "deployment_checklist": {
            "jwt_secret_custom": True,
            "dev_mode_off": False,
            "encryption_on": True,
            "rate_limiting_on": True
        },
        "security_features": [
            "✓ RBAC 权限控制已启用",
            "✓ 审计日志实时记录",
            "✓ 静态数据加密存储",
            "✓ 数据自动脱敏引擎"
        ]
    })
    
# ════════════════════════════════════════════════════════════════
#  ★ 增强 C: 律师注册与认证路由
# ════════════════════════════════════════════════════════════════

@app.route('/lawyer/register', methods=['POST'])
def lawyer_register():
    """
    律师账户注册。
    Body: {
      "name": "张律师",
      "license_number": "京A20230001",
      "firm": "某某律所",
      "specialties": ["劳动用工类","合同纠纷"],
      "phone": "13800138001",
      "password": "安全密码"
    }
    注册后需人工核验 (verified=0)，核验通过后方可接受任务。
    """
    data = request.json or {}
    required = ['name', 'license_number', 'phone', 'password']
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"缺少必填字段：{field}"}), 400

    db = get_db()
    # 检查执照号或手机号是否已注册
    existing = db.execute(
        "SELECT id FROM lawyers WHERE license_number=? OR phone=?",
        (data['license_number'], data['phone'])
    ).fetchone()
    if existing:
        return jsonify({"error": "该执照号或手机号已注册"}), 409

    now_str = datetime.now(timezone.utc).isoformat()
    lawyer_id = str(uuid.uuid4())
    pw_hash = bcrypt.hashpw(data['password'].encode(), bcrypt.gensalt()).decode()
    specialties_json = json.dumps(data.get('specialties', []), ensure_ascii=False)

    db.execute(
        """INSERT INTO lawyers
           (id, name, license_number, firm, specialties, phone, password_hash, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (lawyer_id, data['name'], data['license_number'],
         data.get('firm', ''), specialties_json,
         data['phone'], pw_hash, now_str, now_str)
    )
    db.commit()

    return jsonify({
        "message":   "律师账户注册成功，等待平台核验执照",
        "lawyer_id": lawyer_id,
        "verified":  False,
    }), 201


@app.route('/lawyer/login', methods=['POST'])
def lawyer_login():
    """律师登录，返回律师专用 JWT"""
    data = request.json or {}
    phone = data.get('phone', '').strip()
    password = data.get('password', '')

    if not phone or not password:
        return jsonify({"error": "请提供手机号和密码"}), 400

    db = get_db()
    row = db.execute("SELECT * FROM lawyers WHERE phone=?", (phone,)).fetchone()
    if not row:
        return jsonify({"error": "账号或密码错误"}), 401

    lawyer = dict(row)
    if not bcrypt.checkpw(password.encode(), lawyer['password_hash'].encode()):
        return jsonify({"error": "账号或密码错误"}), 401

    if not lawyer['verified']:
        return jsonify({"error": "账号尚未通过核验，请联系平台管理员"}), 403

    token = issue_lawyer_token(lawyer['id'])
    return jsonify({
        "token":   token,
        "name":    lawyer['name'],
        "firm":    lawyer['firm'],
        "rating":  lawyer['rating'],
        "completed_reviews": lawyer['completed_reviews'],
    })


@app.route('/lawyer/pending', methods=['GET'])
@require_lawyer_auth
def lawyer_pending_reviews():
    """律师查看待分配或已分配给自己的待审任务"""
    lawyer_id = g.lawyer_id
    db = get_db()

    # 获取该律师的专长，优先推送匹配分类的任务
    lawyer = db.execute("SELECT specialties FROM lawyers WHERE id=?", (lawyer_id,)).fetchone()
    specialties = json.loads(lawyer['specialties'] or '[]') if lawyer else []

    # 待分配任务（pending）+ 已分配给本律师的任务
    rows = db.execute("""
        SELECT id, category, risk_score, status, priority, created_at, user_notes
        FROM lawyer_reviews
        WHERE (status='pending' OR (status='assigned' AND lawyer_id=?))
        ORDER BY
            CASE priority WHEN 'critical' THEN 0 WHEN 'urgent' THEN 1 ELSE 2 END,
            risk_score DESC,
            created_at ASC
        LIMIT 20
    """, (lawyer_id,)).fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item['specialty_match'] = item['category'] in specialties
        result.append(item)

    return jsonify({"reviews": result, "total": len(result)})


# ════════════════════════════════════════════════════════════════
#  ★ 增强 C: 用户申请/查看律师审查路由
# ════════════════════════════════════════════════════════════════

@app.route('/review/request', methods=['POST'])
@require_auth
def review_request():
    """
    用户申请律师审查。
    Body: {
      "contract_text": "...",       （必填）合同原文
      "category": "劳动用工类",
      "ai_result": {...},            （可选）已有AI分析结果
      "risk_score": 82,
      "notes": "需要重点关注第3条",
      "priority": "normal/urgent/critical"
    }
    返回 review_id 和审查任务状态。
    """
    user_id = g.user_id
    data = request.json or {}

    contract_text = data.get('contract_text', '').strip()
    if not contract_text:
        return jsonify({"error": "请提供合同原文"}), 400

    category   = data.get('category', '其他类')
    risk_score = data.get('risk_score', 0)
    priority   = data.get('priority', 'normal')
    notes      = data.get('notes', '')
    ai_result  = data.get('ai_result', {})

    # 根据风险分自动升级优先级
    if risk_score >= 85 and priority == 'normal':
        priority = 'urgent'

    now_str   = datetime.now(timezone.utc).isoformat()
    review_id = str(uuid.uuid4())

    db = get_db()
    db.execute(
        """INSERT INTO lawyer_reviews
           (id, user_id, category, risk_score, contract_text,
            ai_result_json, status, priority, user_notes, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (review_id, user_id, category, risk_score, contract_text,
         json.dumps(ai_result, ensure_ascii=False) if ai_result else None,
         'pending', priority, notes, now_str, now_str)
    )
    db.commit()

    return jsonify({
        "review_id": review_id,
        "status":    "pending",
        "priority":  priority,
        "message":   "审查申请已提交，专业律师将在工作日内与您联系",
        "estimated_hours": {"normal": 24, "urgent": 8, "critical": 2}.get(priority, 24),
    }), 201


@app.route('/review/list', methods=['GET'])
@require_auth
def review_list():
    """用户查看自己的所有律师审查记录"""
    user_id = g.user_id
    db = get_db()
    rows = db.execute("""
        SELECT lr.id, lr.category, lr.risk_score, lr.status, lr.priority,
               lr.endorsement, lr.created_at, lr.completed_at,
               l.name as lawyer_name, l.firm as lawyer_firm, l.rating as lawyer_rating
        FROM lawyer_reviews lr
        LEFT JOIN lawyers l ON lr.lawyer_id = l.id
        WHERE lr.user_id = ?
        ORDER BY lr.created_at DESC
    """, (user_id,)).fetchall()

    return jsonify({
        "reviews": [dict(r) for r in rows],
        "total":   len(rows),
    })


@app.route('/review/<review_id>', methods=['GET'])
@require_auth
def review_detail(review_id):
    """获取具体审查记录详情（包含律师意见）"""
    user_id = g.user_id
    db = get_db()
    row = db.execute("""
        SELECT lr.*, l.name as lawyer_name, l.firm as lawyer_firm,
               l.license_number, l.rating as lawyer_rating, l.specialties as lawyer_specialties
        FROM lawyer_reviews lr
        LEFT JOIN lawyers l ON lr.lawyer_id = l.id
        WHERE lr.id = ? AND lr.user_id = ?
    """, (review_id, user_id)).fetchone()

    if not row:
        return jsonify({"error": "审查记录不存在"}), 404

    result = dict(row)
    # 解析 JSON 字段
    for field in ('ai_result_json', 'endorsed_result_json'):
        if result.get(field):
            try:
                result[field] = json.loads(result[field])
            except Exception:
                pass
    return jsonify(result)


@app.route('/review/<review_id>/assign', methods=['POST'])
@require_lawyer_auth
def review_assign(review_id):
    """律师接受审查任务（状态：pending → assigned）"""
    lawyer_id = g.lawyer_id
    db = get_db()

    # 校验任务状态
    row = db.execute(
        "SELECT status FROM lawyer_reviews WHERE id=?", (review_id,)
    ).fetchone()
    if not row:
        return jsonify({"error": "审查任务不存在"}), 404
    if row['status'] != 'pending':
        return jsonify({"error": f"任务当前状态为 '{row['status']}'，无法接受"}), 409

    now_str = datetime.now(timezone.utc).isoformat()
    db.execute(
        "UPDATE lawyer_reviews SET lawyer_id=?, status='assigned', updated_at=? WHERE id=?",
        (lawyer_id, now_str, review_id)
    )
    db.commit()
    return jsonify({"message": "任务已接受，请在规定时间内提交审查意见", "review_id": review_id})


@app.route('/review/<review_id>/submit', methods=['POST'])
@require_lawyer_auth
def review_submit(review_id):
    """
    律师提交专业审查意见（状态：assigned/in_review → completed）。
    Body: {
      "lawyer_opinion": "详细专业意见...",
      "lawyer_notes": "内部备注（用户不可见）",
      "endorsement": "endorsed / amended / overridden",
      "endorsed_result": {...}   （endorsement=amended/overridden 时必须提供修订后结果）
    }
    """
    lawyer_id = g.lawyer_id
    data = request.json or {}

    db = get_db()
    row = db.execute(
        "SELECT status, lawyer_id FROM lawyer_reviews WHERE id=?", (review_id,)
    ).fetchone()
    if not row:
        return jsonify({"error": "审查任务不存在"}), 404
    if row['lawyer_id'] != lawyer_id:
        return jsonify({"error": "无权操作此任务"}), 403
    if row['status'] not in ('assigned', 'in_review'):
        return jsonify({"error": f"任务状态 '{row['status']}' 不可提交"}), 409

    opinion    = data.get('lawyer_opinion', '').strip()
    endorsement = data.get('endorsement', 'endorsed')
    if not opinion:
        return jsonify({"error": "请提供律师意见"}), 400
    if endorsement not in ('endorsed', 'amended', 'overridden'):
        return jsonify({"error": "endorsement 取值需为 endorsed / amended / overridden"}), 400

    endorsed_result = data.get('endorsed_result', {})
    endorsed_json = (
        json.dumps(endorsed_result, ensure_ascii=False)
        if endorsed_result else None
    )

    now_str = datetime.now(timezone.utc).isoformat()
    db.execute(
        """UPDATE lawyer_reviews
           SET status='completed', lawyer_opinion=?, lawyer_notes=?,
               endorsement=?, endorsed_result_json=?,
               completed_at=?, updated_at=?
           WHERE id=?""",
        (opinion, data.get('lawyer_notes', ''), endorsement,
         endorsed_json, now_str, now_str, review_id)
    )
    # 更新律师完成计数
    db.execute(
        "UPDATE lawyers SET completed_reviews=completed_reviews+1, updated_at=? WHERE id=?",
        (now_str, lawyer_id)
    )
    db.commit()

    return jsonify({
        "message":     "审查意见已提交",
        "review_id":   review_id,
        "endorsement": endorsement,
    })


# ════════════════════════════════════════════════════════════════
#  企业级鉴权 API（用户侧，与原版保持一致）
# ════════════════════════════════════════════════════════════════

@app.route('/auth/send-otp', methods=['POST'])
def auth_send_otp():
    data = request.json or {}
    phone = data.get('phone', '').strip()
    if not phone or not re.match(r'^1[3-9]\d{9}$', phone):
        return jsonify({"error": "请提供有效的手机号码"}), 400
    try:
        _, dev_hint = _otp_generate(phone)
        resp = {"message": f"验证码已发送至 {phone[:3]}****{phone[-4:]}"}
        if DEV_MODE and dev_hint:
            resp["dev_otp"] = dev_hint
        return jsonify(resp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 429

@app.route('/auth/login-sms', methods=['POST'])
@app.route('/auth/verify-otp', methods=['POST'])
def auth_verify_otp():
    data  = request.json or {}
    phone = data.get('phone', '').strip()
    code  = data.get('code', '').strip()
    if not phone or not code:
        return jsonify({"error": "请提供手机号和验证码"}), 400
    try:
        _otp_verify(phone, code)
    except ValueError as e:
        return jsonify({"error": str(e)}), 401

    db   = get_db()
    user = _db_get_user_by_phone(db, phone)
    if not user:
        user = _db_create_user(db, phone)
    else:
        user = _user_to_dict(db.execute("SELECT * FROM users WHERE phone=?", (phone,)).fetchone())

    access_token  = issue_access_token(user['id'])
    refresh_token = issue_refresh_token(user['id'])
    return jsonify({"access_token": access_token, "refresh_token": refresh_token, "user": user})


@app.route('/auth/refresh', methods=['POST'])
def auth_refresh():
    data = request.json or {}
    token = data.get('refresh_token', '')
    if not token:
        return jsonify({"error": "missing_refresh_token"}), 400
    try:
        payload = verify_token(token, "refresh")
    except pyjwt.ExpiredSignatureError:
        return jsonify({"error": "refresh_token_expired"}), 401
    except pyjwt.InvalidTokenError as e:
        return jsonify({"error": str(e)}), 401

    revoke_token(token)
    user_id = payload["sub"]
    return jsonify({
        "access_token":  issue_access_token(user_id),
        "refresh_token": issue_refresh_token(user_id),
    })


@app.route('/auth/logout', methods=['POST'])
@require_auth
def auth_logout():
    auth_header = request.headers.get("Authorization", "")
    revoke_token(auth_header[7:])
    return jsonify({"message": "已成功登出"})


@app.route('/auth/me', methods=['GET'])
@require_auth
def auth_me():
    db   = get_db()
    user = _db_get_user_by_id(db, g.user_id)
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify(user)


@app.route('/auth/me', methods=['PUT'])
@require_auth
def auth_update_me():
    data    = request.json or {}
    user_id = g.user_id
    allowed = ['nickname', 'email', 'bio', 'notifications']
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify({"error": "没有可更新的字段"}), 400
    if 'notifications' in updates and isinstance(updates['notifications'], dict):
        updates['notifications'] = json.dumps(updates['notifications'])
    now_str = datetime.now(timezone.utc).isoformat()
    updates['updated_at'] = now_str
    db = get_db()
    cols = ', '.join([f"{k}=?" for k in updates])
    db.execute(f"UPDATE users SET {cols} WHERE id=?", list(updates.values()) + [user_id])
    db.commit()
    return jsonify(_db_get_user_by_id(db, user_id))


@app.route('/auth/contracts', methods=['GET'])
@require_auth
def auth_get_contracts():
    user_id = g.user_id
    page    = max(1, int(request.args.get('page', 1)))
    per     = min(50, max(1, int(request.args.get('per_page', 10))))
    offset  = (page - 1) * per
    db      = get_db()

    total = db.execute("SELECT COUNT(*) FROM contracts WHERE user_id=?", (user_id,)).fetchone()[0]
    rows  = db.execute(
        """SELECT id, date, category, contract_type, risk_score, overall_risk,
                  summary, jurisdiction, confidence_score, models_used, 
                  review_requested, created_at, issues
           FROM contracts WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?""",
        (user_id, per, offset)
    ).fetchall()

    contracts = []
    for r in rows:
        item = dict(r)
        try:
            if item.get('models_used'):
                item['models_used'] = json.loads(item['models_used'])
            if item.get('issues'):
                item['issues'] = json.loads(item['issues'])
            else:
                item['issues'] = []
        except Exception:
            item['issues'] = []
            item['models_used'] = []
                
        contracts.append(item)

    return jsonify({
        "contracts": contracts,
        "total":    total,
        "page":     page,
        "per_page": per,
        "pages":    (total + per - 1) // per,
    })


@app.route('/auth/contracts', methods=['POST'])
@require_auth
def auth_save_contract():
    user_id = g.user_id
    data    = request.json or {}
    now_str = datetime.now(timezone.utc).isoformat()
    cid     = str(uuid.uuid4())

    issues_json   = json.dumps(data.get('issues', []),  ensure_ascii=False)
    models_json   = json.dumps(data.get('models_used', []), ensure_ascii=False)

    db = get_db()
    db.execute(
        """INSERT INTO contracts
           (id, user_id, date, category, contract_type, risk_score, overall_risk,
            summary, jurisdiction, issues, confidence_score, models_used, review_requested, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, user_id, data.get('date', now_str[:10]),
         data.get('category', ''), 
         data.get('contract_type', data.get('contractType', '')), 
         data.get('risk_score', data.get('riskScore', 0)), 
         data.get('overall_risk', data.get('overallRisk', '')),
         data.get('summary', ''), data.get('jurisdiction', ''),
         issues_json,
         data.get('_overall_confidence'),
         models_json,
         int(data.get('review_requested', False)),
         now_str)
    )
    db.execute("UPDATE users SET review_count=review_count+1 WHERE id=?", (user_id,))
    db.commit()
    return jsonify({"id": cid, "message": "合同记录已保存"}), 201


@app.route('/auth/contracts/<contract_id>', methods=['GET'])
@require_auth
def auth_get_contract(contract_id):
    db  = get_db()
    row = db.execute(
        "SELECT * FROM contracts WHERE id=? AND user_id=?", (contract_id, g.user_id)
    ).fetchone()
    if not row:
        return jsonify({"error": "合同不存在"}), 404
    d = dict(row)
    for field in ('issues', 'models_used'):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except Exception:
                pass
    return jsonify(d)


@app.route('/auth/contracts/<contract_id>', methods=['DELETE'])
@require_auth
def auth_delete_contract(contract_id):
    db  = get_db()
    row = db.execute(
        "SELECT id FROM contracts WHERE id=? AND user_id=?", (contract_id, g.user_id)
    ).fetchone()
    if not row:
        return jsonify({"error": "合同不存在"}), 404
    db.execute("DELETE FROM contracts WHERE id=?", (contract_id,))
    db.execute("UPDATE users SET review_count=MAX(0,review_count-1) WHERE id=?", (g.user_id,))
    db.commit()
    return jsonify({"message": "已删除"})


@app.route('/auth/favorites', methods=['GET'])
@require_auth
def auth_get_favorites():
    db   = get_db()
    rows = db.execute("""
        SELECT c.id, c.date, c.category, c.contract_type, c.risk_score, c.overall_risk,
               c.summary, f.created_at as favorited_at
        FROM favorites f JOIN contracts c ON f.contract_id=c.id
        WHERE f.user_id=? ORDER BY f.created_at DESC
    """, (g.user_id,)).fetchall()
    return jsonify({"favorites": [dict(r) for r in rows]})


@app.route('/auth/favorites/<contract_id>', methods=['POST'])
@require_auth
def auth_add_favorite(contract_id):
    db = get_db()
    if not db.execute("SELECT id FROM contracts WHERE id=? AND user_id=?",
                      (contract_id, g.user_id)).fetchone():
        return jsonify({"error": "合同不存在"}), 404
    try:
        db.execute(
            "INSERT INTO favorites (user_id, contract_id, created_at) VALUES (?,?,?)",
            (g.user_id, contract_id, datetime.now(timezone.utc).isoformat())
        )
        db.commit()
    except sqlite3.IntegrityError:
        pass
    return jsonify({"message": "已添加收藏"})


@app.route('/auth/favorites/<contract_id>', methods=['DELETE'])
@require_auth
def auth_remove_favorite(contract_id):
    db = get_db()
    db.execute("DELETE FROM favorites WHERE user_id=? AND contract_id=?",
               (g.user_id, contract_id))
    db.commit()
    return jsonify({"message": "已取消收藏"})


@app.route('/auth/stats', methods=['GET'])
@require_auth
def auth_get_stats():
    user_id = g.user_id
    db = get_db()
    monthly = db.execute("""
        SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as cnt
        FROM contracts WHERE user_id=? AND created_at >= date('now', '-6 months')
        GROUP BY month ORDER BY month ASC
    """, (user_id,)).fetchall()
    risk_dist = db.execute(
        "SELECT overall_risk, COUNT(*) as cnt FROM contracts WHERE user_id=? GROUP BY overall_risk",
        (user_id,)
    ).fetchall()
    by_cat = db.execute("""
        SELECT category, ROUND(AVG(risk_score),1) as avg_score, COUNT(*) as cnt
        FROM contracts WHERE user_id=? GROUP BY category ORDER BY cnt DESC LIMIT 6
    """, (user_id,)).fetchall()
    top = db.execute("""
        SELECT contract_type, risk_score, category
        FROM contracts WHERE user_id=? ORDER BY risk_score DESC LIMIT 3
    """, (user_id,)).fetchall()

    return jsonify({
        'monthly_reviews':   [{'month': r['month'], 'count': r['cnt']} for r in monthly],
        'risk_distribution': {r['overall_risk']: r['cnt'] for r in risk_dist if r['overall_risk']},
        'by_category':       [{'category': r['category'], 'avg_score': r['avg_score'], 'count': r['cnt']} for r in by_cat],
        'top_risk':          [{'contract_type': r['contract_type'], 'risk_score': r['risk_score'], 'category': r['category']} for r in top],
    })


@app.route('/auth/contracts/batch-delete', methods=['POST'])
@require_auth
def auth_batch_delete_contracts():
    data = request.json or {}
    ids  = data.get('ids', [])
    if not ids or not isinstance(ids, list):
        return jsonify({'error': '请提供要删除的合同ID列表'}), 400
    if len(ids) > 50:
        return jsonify({'error': '单次最多删除50条记录'}), 400

    db = get_db()
    placeholders = ','.join(['?' for _ in ids])
    params = ids + [g.user_id]
    rows = db.execute(
        f"SELECT id FROM contracts WHERE id IN ({placeholders}) AND user_id=?", params
    ).fetchall()
    valid_ids = [r['id'] for r in rows]
    if not valid_ids:
        return jsonify({'error': '未找到可删除的记录'}), 404

    ph2 = ','.join(['?' for _ in valid_ids])
    db.execute(f"DELETE FROM contracts WHERE id IN ({ph2})", valid_ids)
    db.execute(
        "UPDATE users SET review_count=MAX(0,review_count-?) WHERE id=?",
        (len(valid_ids), g.user_id)
    )
    db.commit()
    return jsonify({'message': f'已成功删除 {len(valid_ids)} 条记录', 'deleted': len(valid_ids)})


# ════════════════════════════════════════════════════════════════
#  系统监控路由
# ════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Service is live"}), 200


@app.route('/cache/clear', methods=['POST'])
@require_auth
def cache_clear():
    data   = request.json or {}
    hours  = int(data.get('older_than_hours', CACHE_TTL_HOURS))
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    db = get_db()
    result = db.execute("DELETE FROM analysis_cache WHERE created_at < ?", (cutoff,))
    db.commit()
    return jsonify({"message": f"已清理 {result.rowcount} 条过期缓存（>{hours}h）"})


@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        total  = conn.execute("SELECT COUNT(*) as n, SUM(hit_count) as hits FROM analysis_cache").fetchone()
        by_cat = conn.execute(
            "SELECT category, COUNT(*) as n, SUM(hit_count) as hits FROM analysis_cache GROUP BY category"
        ).fetchall()
        conn.close()
        return jsonify({
            'total_entries': total['n'] or 0,
            'total_hits':    total['hits'] or 0,
            'by_category':   [dict(r) for r in by_cat],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════
#  后台线程：任务 GC
# ════════════════════════════════════════════════════════════════

def _task_gc_worker():
    """每 30 分钟回收已完成/失败且超过 2 小时的任务对象"""
    while True:
        time.sleep(1800)
        now = time.time()
        to_delete = [
            tid for tid, t in list(tasks.items())
            if t.get('status') in ('completed', 'failed')
            and now - t.get('_completed_at', now) > 7200
        ]
        for tid in to_delete:
            tasks.pop(tid, None)
        if to_delete:
            print(f"[GC] 回收 {len(to_delete)} 个过期任务对象")



# ════════════════════════════════════════════════════════════════
#  ★ 新增: SSE 流式分析端点（Fast & Slow 架构 + 流式输出）
# ════════════════════════════════════════════════════════════════

def _do_quick_scan_sync(contract_text: str, category: str, language: str, party_role: str) -> dict:
    """同步执行快速风险扫描（用于 SSE 流的第一阶段，目标 10 秒内返回）"""
    lang_name      = LANGUAGE_MAP.get(language, 'Simplified Chinese (简体中文)')
    party_label    = "甲方" if party_role == 'partyA' else "乙方"
    party_opposite = "乙方" if party_role == 'partyA' else "甲方"

    # 使用更小的 max_tokens 加速响应
    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com",
        max_tokens=1200,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    docs, doc_count, _ = get_law_docs_enhanced(contract_text, category, k=3, include_cases=False)
    laws_brief = "\n".join([
        f"[{d.metadata.get('law_name','?')}]: {d.page_content[:200]}"
        for d in docs[:3]
    ])

    if category == '网络数字类':
        prompt = f"""你是互联网法律合规专家。快速扫描以下用户协议/隐私政策，
识别对用户（{party_label}）最重要的 3-5 个风险点或权益授权问题。
【合同分类】：{category}（网络平台用户协议/隐私政策）
【参考法条摘要】：{laws_brief or '（无检索结果）'}
【协议内容（前4000字）】：{contract_text[:4000]}

重点关注：①过宽的数据收集授权 ②模糊的第三方共享 ③不合理的免责条款 ④缺失的用户权益保障 ⑤单方面变更协议条款

输出 JSON（使用 {lang_name}）：
{{
  "quickRiskLevel": "极高/高/中/低",
  "quickScore": 整数0-100,
  "contractType": "推断的合同类型（如：APP用户协议、隐私政策等）",
  "topThreats": [
    {{"title": "风险标题", "severity": "极高/高/中/低",
      "brief": "一句话说明危害", "clauseHint": "涉及条款简述（10字内）"}}
  ],
  "quickTip": "最重要一条建议（20字内）",
  "lawyerReviewSuggested": true
}}"""
    else:
        prompt = f"""你是法律专家。以本合同【{party_label}】的视角，快速扫描以下合同，
识别对【{party_label}】最危险的 3-5 个风险点（优先选高危/极高危）。
【合同分类】：{category}
【审查视角】：{party_label}（对方为{party_opposite}）
【参考法条摘要】：{laws_brief or '（无检索结果）'}
【合同内容（前4000字）】：{contract_text[:4000]}

输出 JSON（使用 {lang_name}）：
{{
  "quickRiskLevel": "极高/高/中/低",
  "quickScore": 整数0-100,
  "contractType": "推断的合同类型",
  "topThreats": [
    {{"title": "风险标题", "severity": "极高/高/中/低",
      "brief": "一句话说明危害", "clauseHint": "涉及条款简述（10字内）"}}
  ],
  "quickTip": "最重要一条建议（20字内）",
  "lawyerReviewSuggested": true
}}"""

    res = _retry_llm(lambda: llm.invoke(prompt), max_attempts=2)
    result = json.loads(robust_json_cleaner(res.content), strict=False)
    result['_law_doc_count'] = doc_count
    result['_party_role']    = party_role
    result['_party_label']   = party_label
    return result


@app.route('/analyze/stream', methods=['POST'])
def analyze_stream():
    """
    ★ SSE 流式分析端点（Fast & Slow 双轨架构）

    事件流结构：
      connected    → 连接建立（含 warnings）
      quick_scan   → 快速风险扫描完成（~10s 内），含 topThreats
      stage1_ready → 深度模型已完成风险识别，含完整 issues 列表
      progress     → 阶段进度更新
      complete     → 全流程完成，含最终 result
      error        → 错误
    """
    try:
        ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
        allowed, remaining = _check_analysis_rate_limit(ip)
        if not allowed:
            def _err():
                yield f"data: {json.dumps({'type':'error','message':f'请求过于频繁，每小时最多 {ANALYSIS_RATE_LIMIT_PER_HOUR} 次'}, ensure_ascii=False)}\n\n"
            return Response(_err(), mimetype='text/event-stream')

        data       = request.json or {}
        raw_text   = data.get('text', '').strip()
        category   = data.get('category', '其他类')
        language   = data.get('language', 'zh-CN')
        party_role = data.get('party_role', 'partyA')
        if party_role not in ('partyA', 'partyB'):
            party_role = 'partyA'
        if language not in LANGUAGE_MAP:
            language = 'zh-CN'

        if not raw_text:
            def _err():
                yield f"data: {json.dumps({'type':'error','message':'无合同内容'}, ensure_ascii=False)}\n\n"
            return Response(_err(), mimetype='text/event-stream')

        try:
            contract_text, warnings = _sanitize_contract_text(raw_text)
        except ValueError as e:
            def _err():
                yield f"data: {json.dumps({'type':'error','message':str(e)}, ensure_ascii=False)}\n\n"
            return Response(_err(), mimetype='text/event-stream')

        event_q = _queue.Queue()

        def emit(event_type: str, payload: dict):
            """向 SSE 队列投递一条事件"""
            event_q.put({'type': event_type, **payload})

        def background():
            task_id  = None
            try:
                # ─── Phase 1: 快速扫描（Fast, ~10s）───────────────
                emit('progress', {'message': '⚡ 初步风险扫描中（通常 10 秒内完成）...', 'stage': 0})
                try:
                    quick = _do_quick_scan_sync(contract_text, category, language, party_role)
                    emit('quick_scan', {'result': quick})
                    emit('progress', {'message': '✅ 风险初筛完毕，正在启动深度多模型审查团...', 'stage': 1})
                except Exception as qe:
                    print(f"[stream] quick scan failed: {qe}")
                    emit('progress', {'message': '正在启动深度审查...', 'stage': 1})

                # ─── 缓存命中检查 ─────────────────────────────────
                text_hash = _contract_hash(contract_text, category, language, party_role)
                cached = _get_cached_result(text_hash, category, language)
                if cached:
                    emit('progress', {'message': '⚡ 命中分析缓存，极速返回结果...', 'stage': 5})
                    emit('stage1_ready', {
                        'issues': cached.get('issues', []),
                        'meta': {
                            'contractType': cached.get('contractType', ''),
                            'overallRisk':  cached.get('overallRisk', ''),
                            'riskScore':    cached.get('riskScore', 0),
                            'summary':      cached.get('summary', ''),
                        }
                    })
                    emit('complete', {'result': cached, 'cached': True})
                    return

                # ─── Phase 2: 深度分析（Slow，后台线程）──────────
                task_id  = str(uuid.uuid4())
                audit_id = str(uuid.uuid4())
                _log_analysis_audit(audit_id, ip, text_hash, category, language)

                tasks[task_id] = {
                    "status":        "processing",
                    "stage":         1,
                    "progress":      "正在组建多模型审查团...",
                    "_stage1_ready": False,
                    "_created_at":   time.time(),
                }

                threading.Thread(
                    target=run_deep_analysis,
                    args=(task_id, contract_text, category, language, audit_id, party_role),
                    daemon=True,
                ).start()

                # ─── 进度监听循环 ──────────────────────────────────
                stage1_pushed = False
                last_progress = ''

                while True:
                    task = tasks.get(task_id, {})
                    status = task.get('status', 'processing')
                    prog   = task.get('progress', '')
                    stage  = task.get('stage', 1)

                    # 进度文本变化时推送
                    if prog and prog != last_progress:
                        emit('progress', {'message': prog, 'stage': stage})
                        last_progress = prog

                    # Stage 1 完成后立即推送风险清单
                    if not stage1_pushed and task.get('_stage1_ready'):
                        stage1_pushed = True
                        emit('stage1_ready', {
                            'issues': task.get('_stage1_issues', []),
                            'meta':   task.get('_stage1_meta', {}),
                        })

                    if status == 'completed':
                        result = task.get('result', {})
                        emit('complete', {'result': result, 'cached': False})
                        break
                    elif status == 'failed':
                        emit('error', {'message': task.get('error', '分析失败')})
                        break

                    time.sleep(0.4)

            except Exception as e:
                print(f"[stream] background error: {traceback.format_exc()}")
                emit('error', {'message': str(e)})
            finally:
                event_q.put(None)  # 结束信号

        threading.Thread(target=background, daemon=True).start()

        @stream_with_context
        def generate():
            yield f"data: {json.dumps({'type':'connected','warnings': warnings}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    item = event_q.get(timeout=150)
                    if item is None:
                        yield f"data: {json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"
                        break
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except _queue.Empty:
                    # heartbeat — 防止反代超时断连
                    yield ": heartbeat\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control':      'no-cache',
                'X-Accel-Buffering':  'no',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,bypass-tunnel-reminder',
            }
        )

    except Exception as e:
        def _err():
            yield f"data: {json.dumps({'type':'error','message':str(e)}, ensure_ascii=False)}\n\n"
        return Response(_err(), mimetype='text/event-stream')


if __name__ == '__main__':
    print("正在启动 ContractClarity 专家审查引擎 v3.0...")
    threading.Thread(target=_task_gc_worker, daemon=True).start()
    print("✓ 任务 GC 后台线程已启动")
    print(f"✓ 多模型交叉验证：{'已启用（' + str(len(_get_available_models())) + '个模型）' if ENABLE_MULTI_MODEL else '未启用（单模型）'}")
    print(f"✓ 律师介入阈值：风险分 ≥ {AUTO_SUGGEST_REVIEW_SCORE}")
    print(f"✓ 分析频率限制：{ANALYSIS_RATE_LIMIT_PER_HOUR} 次/小时/IP")
    print(f"✓ 分析结果缓存 TTL：{CACHE_TTL_HOURS} 小时")
    print(f"✓ LLM 最大重试：{LLM_MAX_RETRIES} 次")
    print("✓ ContractClarity v3.0 后端已就绪。")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
