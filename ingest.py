"""
ingest.py  ──  ContractClarity 法律向量库构建脚本（动态知识库版 v2.0）

★ 新增增强（在 v1.0 基础上）：

增强 B: 动态法律知识库
─────────────────────────────────────────────────────────────────
1. 多文档类型支持
   新增四类文档的差异化入库策略：
   · 法律法规（原有）    ── 大字号切块 chunk_size=600
   · 司法解释            ── 中等切块 chunk_size=500，保留裁判要旨完整性
   · 典型案例/裁判文书   ── 专项向量库 chroma_db/典型案例，供类案检索
   · 部门规章/地方标准   ── 并入所属分类库，但标注 document_type=部门规章

2. 知识库版本清单（kb_manifest.json）
   每次构建后自动更新：
   · last_updated       上次更新时间
   · total_laws         入库法律文件总数
   · total_chunks        入库文本块总数
   · category_stats     各分类详细统计（法律数、块数、文档类型分布）
   · schema_version     便于 app.py 判断知识库是否兼容当前版本

3. 有效期检测与警告
   文件名中包含 [废止] / [失效] 字样时自动跳过，并记录警告日志
   文件名中包含年份时提取为 effective_year 元数据

4. 独立案例库构建
   ./法律条文/典型案例/ 目录下的文件构建独立的 chroma_db/典型案例 向量库
   案例文档使用更小的切块策略（chunk_size=400）以保留案情细节
   案例元数据额外注入：case_result（如："支持"/"驳回"）通过文件名推断

5. 司法解释优先检索标记
   司法解释类文档在元数据中标注 priority=high，检索时可权重加分

目录结构约定（v2.0）：
   ./法律条文/
   ├── 劳动用工类/          ── 分类专属法律条文（法律法规 + 部门规章）
   ├── 房产物业类/
   ├── ...（其他分类）
   ├── 通用/                ── 民法典等基础法（并入所有分类库）
   ├── 司法解释/            ── 最高法/最高检司法解释（并入对应分类库）
   │   ├── 劳动用工类/
   │   ├── 通用/
   │   └── ...
   └── 典型案例/            ── 裁判文书/典型案例（独立案例库）
       ├── 劳动用工类/
       ├── 房产物业类/
       └── ...
"""

import os
import json
import time
import hashlib
import shutil
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ─────────────────── 目录与路径配置 ───────────────────
SOURCE_LAWS_DIR     = './法律条文'
DB_BASE_DIR         = './chroma_db'
HASH_STORE_PATH     = './chroma_db/.file_hashes.json'
KB_MANIFEST_PATH    = './chroma_db/kb_manifest.json'   # ★ 知识库版本清单
KB_SCHEMA_VERSION   = '2.0'                             # 版本号，供 app.py 兼容性检查

# ★ 新增：专项子目录
JUDICIAL_INTERP_DIR = '司法解释'   # 司法解释子目录（相对于 SOURCE_LAWS_DIR）
CASES_DIR           = '典型案例'   # 典型案例子目录
CASES_DB_NAME       = '典型案例'   # 独立案例向量库名称

LAW_CATEGORIES = [
    '劳动用工类',
    '房产物业类',
    '消费服务类',
    '金融借贷类',
    '网络数字类',
    '婚姻家庭类',
    '经营合作类',
    '其他类',
]
GENERAL_CATEGORY = '通用'

# ─────────────────── 文档类型定义 ───────────────────
DOCUMENT_TYPE_LAW         = '法律法规'
DOCUMENT_TYPE_JUDICIAL    = '司法解释'
DOCUMENT_TYPE_CASE        = '典型案例'
DOCUMENT_TYPE_REGULATION  = '部门规章'

# 各文档类型对应的切块参数
CHUNK_CONFIGS = {
    DOCUMENT_TYPE_LAW: {
        'chunk_size': 600, 'chunk_overlap': 80,
        'description': '法律法规（标准切块）'
    },
    DOCUMENT_TYPE_JUDICIAL: {
        'chunk_size': 500, 'chunk_overlap': 100,
        'description': '司法解释（保留裁判要旨）'
    },
    DOCUMENT_TYPE_CASE: {
        'chunk_size': 400, 'chunk_overlap': 60,
        'description': '典型案例（保留案情细节）'
    },
    DOCUMENT_TYPE_REGULATION: {
        'chunk_size': 600, 'chunk_overlap': 80,
        'description': '部门规章（标准切块）'
    },
}

# 通用切块分隔符（中文优化）
SEPARATORS = ['\n\n\n', '\n\n', '\n', '。', '；', '，', '　', ' ', '']

VALIDATION_QUERY = "违约责任如何认定"
MAX_WORKERS      = 4

# 文件名废止关键词（自动跳过）
OBSOLETE_KEYWORDS = ['废止', '失效', '已废除', '已失效', '作废']


# ─────────────────── 工具函数 ───────────────────

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_hash_store() -> dict:
    if os.path.exists(HASH_STORE_PATH):
        try:
            with open(HASH_STORE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_hash_store(store: dict):
    os.makedirs(os.path.dirname(HASH_STORE_PATH), exist_ok=True)
    with open(HASH_STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _load_kb_manifest() -> dict:
    """加载知识库版本清单"""
    if os.path.exists(KB_MANIFEST_PATH):
        try:
            with open(KB_MANIFEST_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'schema_version': KB_SCHEMA_VERSION,
        'last_updated': None,
        'total_laws': 0,
        'total_chunks': 0,
        'category_stats': {},
    }


def _save_kb_manifest(manifest: dict):
    """持久化知识库版本清单"""
    os.makedirs(os.path.dirname(KB_MANIFEST_PATH), exist_ok=True)
    with open(KB_MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"   📋 知识库清单已更新：{KB_MANIFEST_PATH}")


def _extract_law_name(file_path: str) -> str:
    stem = Path(file_path).stem
    return stem.replace('_', ' ').replace('-', ' ')


def _is_obsolete(file_path: str) -> bool:
    """检测文件是否为已废止/失效的法律，是则跳过"""
    name = Path(file_path).stem
    return any(kw in name for kw in OBSOLETE_KEYWORDS)


def _extract_year_from_name(file_path: str) -> str | None:
    """从文件名提取年份（如 '劳动合同法2008.txt' → '2008'）"""
    match = re.search(r'(19|20)\d{2}', Path(file_path).stem)
    return match.group(0) if match else None


def _infer_document_type(file_path: str, source_dir_hint: str = None) -> str:
    """
    根据文件路径和源目录推断文档类型。
    优先级：源目录 hint > 文件名关键词 > 默认法律法规
    """
    if source_dir_hint:
        if JUDICIAL_INTERP_DIR in source_dir_hint:
            return DOCUMENT_TYPE_JUDICIAL
        if CASES_DIR in source_dir_hint:
            return DOCUMENT_TYPE_CASE

    stem = Path(file_path).stem
    judicial_keywords = ['司法解释', '解释', '批复', '意见', '规定', '复函']
    case_keywords     = ['案例', '判决', '裁定', '裁判', '典型案件']
    regulation_keywords = ['办法', '规程', '条例', '规范', '标准', '通知']

    for kw in case_keywords:
        if kw in stem:
            return DOCUMENT_TYPE_CASE
    for kw in judicial_keywords:
        if kw in stem:
            return DOCUMENT_TYPE_JUDICIAL
    for kw in regulation_keywords:
        if kw in stem:
            return DOCUMENT_TYPE_REGULATION

    return DOCUMENT_TYPE_LAW


def _infer_case_result(file_path: str) -> str:
    """
    从文件名推断案例裁判结果（用于案例库元数据）。
    示例：'劳动者胜诉_违法解除.txt' → '支持'
    """
    stem = Path(file_path).stem.lower()
    if any(kw in stem for kw in ['胜诉', '支持', '获赔', '认定']):
        return '支持原告'
    if any(kw in stem for kw in ['败诉', '驳回', '不予支持']):
        return '驳回'
    return '未知'


def load_documents_with_metadata(
    dir_path: str,
    category: str,
    doc_type: str = DOCUMENT_TYPE_LAW,
    source_dir_hint: str = None,
) -> list:
    """
    加载目录下所有 .txt 文档，注入富元数据（v2.0 增强）。
    新增字段：document_type / effective_year / priority / case_result
    自动跳过已废止文件。
    """
    if not os.path.exists(dir_path):
        return []

    all_docs = []
    txt_files = list(Path(dir_path).glob('**/*.txt'))
    skipped_obsolete = 0

    for txt_file in txt_files:
        # ★ 废止检测
        if _is_obsolete(str(txt_file)):
            print(f"   ⚠ 跳过已废止文件：{txt_file.name}")
            skipped_obsolete += 1
            continue

        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            raw_docs = loader.load()
            law_name = _extract_law_name(str(txt_file))
            file_hash = _file_sha256(str(txt_file))

            # ★ 推断文档类型
            actual_doc_type = _infer_document_type(
                str(txt_file), source_dir_hint=source_dir_hint or str(txt_file)
            ) if doc_type == DOCUMENT_TYPE_LAW else doc_type

            # ★ 提取年份
            effective_year = _extract_year_from_name(str(txt_file))

            # ★ 优先级标注
            priority = 'high' if actual_doc_type == DOCUMENT_TYPE_JUDICIAL else 'normal'

            # ★ 案例裁判结果
            case_result = _infer_case_result(str(txt_file)) \
                if actual_doc_type == DOCUMENT_TYPE_CASE else None

            for doc in raw_docs:
                doc.metadata.update({
                    'source_file':   str(txt_file),
                    'law_name':      law_name,
                    'category':      category,
                    'document_type': actual_doc_type,
                    'file_hash':     file_hash,
                    'ingested_at':   datetime.now().isoformat(),
                    'priority':      priority,
                })
                if effective_year:
                    doc.metadata['effective_year'] = effective_year
                if case_result:
                    doc.metadata['case_result'] = case_result

            all_docs.extend(raw_docs)
        except Exception as e:
            print(f"   ⚠ 加载失败（{txt_file.name}）: {e}")

    if skipped_obsolete:
        print(f"   🚫 跳过废止文件 {skipped_obsolete} 个")

    return all_docs


def _make_splitter(doc_type: str) -> RecursiveCharacterTextSplitter:
    """根据文档类型创建对应的文本分割器"""
    cfg = CHUNK_CONFIGS.get(doc_type, CHUNK_CONFIGS[DOCUMENT_TYPE_LAW])
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg['chunk_size'],
        chunk_overlap=cfg['chunk_overlap'],
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def split_with_metadata(docs: list, category: str) -> list:
    """
    ★ v2.0 增强：按文档类型分组后差异化切分，保留富元数据。
    """
    # 按文档类型分组
    type_groups: dict[str, list] = {}
    for doc in docs:
        dtype = doc.metadata.get('document_type', DOCUMENT_TYPE_LAW)
        type_groups.setdefault(dtype, []).append(doc)

    all_split = []
    for dtype, group_docs in type_groups.items():
        splitter = _make_splitter(dtype)
        split_docs = splitter.split_documents(group_docs)
        cfg = CHUNK_CONFIGS.get(dtype, CHUNK_CONFIGS[DOCUMENT_TYPE_LAW])
        print(f"   ✂ [{dtype}] {len(group_docs)} 文档 → {len(split_docs)} 块"
              f"（chunk_size={cfg['chunk_size']}）")
        all_split.extend(split_docs)

    # 追踪 chunk_index（按来源文件）
    chunk_counters: dict = {}
    for doc in all_split:
        src = doc.metadata.get('source_file', 'unknown')
        idx = chunk_counters.get(src, 0)
        doc.metadata['chunk_index'] = idx
        chunk_counters[src] = idx + 1

    return all_split


def deduplicate_chunks(chunks: list) -> list:
    seen: set = set()
    unique: list = []
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(chunk)
    duplicates = len(chunks) - len(unique)
    if duplicates:
        print(f"   ♻ 去除重复块 {duplicates} 个")
    return unique


def validate_vector_db(persist_dir: str, embeddings) -> bool:
    try:
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        results = db.similarity_search(VALIDATION_QUERY, k=2)
        if not results:
            print(f"   ⚠ 验证警告：查询 '{VALIDATION_QUERY}' 无返回结果")
            return False
        # 展示文档类型分布（验证增强元数据）
        type_counts = {}
        for r in results:
            dtype = r.metadata.get('document_type', '未知')
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        print(f"   ✅ 验证通过（查询命中 {len(results)} 条，类型分布: {type_counts}）")
        return True
    except Exception as e:
        print(f"   ✗ 验证失败: {e}")
        return False


def safe_remove_dir(dir_path: str):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"   - 已清理旧向量库: {dir_path}")
        except Exception as e:
            raise RuntimeError(f"删除目录失败（请检查权限）: {e}")


# ─────────────────── ★ 新增：独立案例库构建 ───────────────────

def process_cases_db(embeddings, hash_store: dict, force_rebuild: bool = False) -> dict:
    """
    ★ 构建独立的典型案例向量库（chroma_db/典型案例）。
    汇总所有分类下的案例文件，构建统一的跨分类案例检索库。
    """
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"▶ 构建独立案例库：【{CASES_DB_NAME}】")

    cases_root = os.path.join(SOURCE_LAWS_DIR, CASES_DIR)
    if not os.path.exists(cases_root):
        print(f"   ⚠ 案例目录不存在：{cases_root}，跳过案例库构建")
        return {'category': CASES_DB_NAME, 'status': 'skipped', 'reason': 'no_cases_dir'}

    persist_dir   = os.path.join(DB_BASE_DIR, CASES_DB_NAME)
    all_txt_files = list(Path(cases_root).glob('**/*.txt'))

    if not all_txt_files:
        print(f"   ⚠ 案例目录下未找到 .txt 文件，跳过")
        return {'category': CASES_DB_NAME, 'status': 'skipped', 'reason': 'no_case_files'}

    # 增量检测
    if not force_rebuild and os.path.exists(persist_dir):
        changed = any(
            _file_sha256(str(f)) != hash_store.get(str(f))
            for f in all_txt_files
        )
        if not changed:
            print(f"   ✓ 案例库无变更，跳过重建")
            return {'category': CASES_DB_NAME, 'status': 'skipped_incremental'}

    # 遍历各分类案例子目录
    all_docs = []
    for subdir in Path(cases_root).iterdir():
        if not subdir.is_dir():
            continue
        cat_name = subdir.name
        cat_docs = load_documents_with_metadata(
            str(subdir), cat_name,
            doc_type=DOCUMENT_TYPE_CASE,
            source_dir_hint=CASES_DIR
        )
        if cat_docs:
            print(f"   + 案例（{cat_name}）：{len(cat_docs)} 个文档")
            all_docs.extend(cat_docs)

    # 根目录下直接放置的案例文件
    root_cases = load_documents_with_metadata(
        cases_root, '通用',
        doc_type=DOCUMENT_TYPE_CASE,
        source_dir_hint=CASES_DIR
    )
    if root_cases:
        print(f"   + 通用案例：{len(root_cases)} 个文档")
        all_docs.extend(root_cases)

    if not all_docs:
        print(f"   ⚠ 所有案例文档加载为空，跳过")
        return {'category': CASES_DB_NAME, 'status': 'skipped', 'reason': 'empty_docs'}

    split_docs = split_with_metadata(all_docs, CASES_DB_NAME)
    split_docs = deduplicate_chunks(split_docs)
    print(f"   ✅ 最终入库案例块数：{len(split_docs)}")

    safe_remove_dir(persist_dir)
    try:
        print(f"   💾 写入案例向量库：{persist_dir} ...")
        Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    except Exception as e:
        print(f"   ✗ 案例向量库写入失败：{e}")
        return {'category': CASES_DB_NAME, 'status': 'failed', 'error': str(e)}

    validate_vector_db(persist_dir, embeddings)

    for f in all_txt_files:
        hash_store[str(f)] = _file_sha256(str(f))

    elapsed = time.time() - t0
    print(f"   ⏱ 案例库构建耗时 {elapsed:.1f}s ✓")
    return {
        'category':    CASES_DB_NAME,
        'status':      'success',
        'doc_count':   len(all_docs),
        'chunk_count': len(split_docs),
        'elapsed_s':   round(elapsed, 1),
    }


# ─────────────────── 核心处理逻辑 ───────────────────

def process_category(
    category: str,
    embeddings,
    hash_store: dict,
    force_rebuild: bool = False,
) -> dict:
    """
    处理单个分类：加载各类型法条 → 切分 → 去重 → 写入向量库 → 验证。
    v2.0 新增：司法解释目录合并、文档类型统计。
    """
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"▶ 处理分类：【{category}】")

    category_doc_path   = os.path.join(SOURCE_LAWS_DIR, category)
    common_doc_path     = os.path.join(SOURCE_LAWS_DIR, GENERAL_CATEGORY)
    # ★ 司法解释路径（分类专属 + 通用）
    judicial_cat_path   = os.path.join(SOURCE_LAWS_DIR, JUDICIAL_INTERP_DIR, category)
    judicial_gen_path   = os.path.join(SOURCE_LAWS_DIR, JUDICIAL_INTERP_DIR, GENERAL_CATEGORY)
    persist_dir         = os.path.join(DB_BASE_DIR, category)

    # ── 收集所有相关文件用于增量检测 ──
    def _glob_txt(p):
        return list(Path(p).glob('**/*.txt')) if os.path.exists(p) else []

    all_txt_files = (
        _glob_txt(category_doc_path) +
        _glob_txt(common_doc_path) +
        _glob_txt(judicial_cat_path) +
        _glob_txt(judicial_gen_path)
    )

    if not all_txt_files:
        print(f"   ⚠ 未找到任何 .txt 文件，跳过分类 [{category}]")
        return {'category': category, 'status': 'skipped', 'reason': 'no_files'}

    if not force_rebuild and os.path.exists(persist_dir):
        changed = any(
            _file_sha256(str(f)) != hash_store.get(str(f))
            for f in all_txt_files
        )
        if not changed:
            print(f"   ✓ 无文件变更，跳过重建（增量缓存）")
            return {'category': category, 'status': 'skipped_incremental'}

    # ── 加载各类型文档 ──
    all_docs = []
    doc_type_counts = {}

    def _load_and_count(dir_path, cat, dtype, hint=None):
        docs = load_documents_with_metadata(dir_path, cat, doc_type=dtype, source_dir_hint=hint)
        if docs:
            doc_type_counts[dtype] = doc_type_counts.get(dtype, 0) + len(docs)
        return docs

    # 1. 分类专属法律条文
    cat_docs = _load_and_count(category_doc_path, category, DOCUMENT_TYPE_LAW)
    all_docs.extend(cat_docs)
    print(f"   + 专属法律条文：{len(cat_docs)} 个文档")

    # 2. 通用法律条文（民法典等）
    common_docs = _load_and_count(common_doc_path, GENERAL_CATEGORY, DOCUMENT_TYPE_LAW)
    all_docs.extend(common_docs)
    print(f"   + 通用法律条文：{len(common_docs)} 个文档")

    # 3. 分类司法解释 ★ 新增
    judicial_cat_docs = _load_and_count(
        judicial_cat_path, category, DOCUMENT_TYPE_JUDICIAL, hint=JUDICIAL_INTERP_DIR
    )
    all_docs.extend(judicial_cat_docs)
    if judicial_cat_docs:
        print(f"   + 专属司法解释：{len(judicial_cat_docs)} 个文档（优先级: HIGH）")

    # 4. 通用司法解释 ★ 新增
    judicial_gen_docs = _load_and_count(
        judicial_gen_path, GENERAL_CATEGORY, DOCUMENT_TYPE_JUDICIAL, hint=JUDICIAL_INTERP_DIR
    )
    all_docs.extend(judicial_gen_docs)
    if judicial_gen_docs:
        print(f"   + 通用司法解释：{len(judicial_gen_docs)} 个文档（优先级: HIGH）")

    if not all_docs:
        print(f"   ⚠ 所有文档加载为空，跳过")
        return {'category': category, 'status': 'skipped', 'reason': 'empty_docs'}

    # ── 按文档类型差异化切分 ──
    split_docs = split_with_metadata(all_docs, category)
    print(f"   ✂ 总计切分为 {len(split_docs)} 个文本块")

    # ── 去重 ──
    split_docs = deduplicate_chunks(split_docs)
    print(f"   ✅ 最终入库块数：{len(split_docs)}")

    # ── 重建向量库 ──
    safe_remove_dir(persist_dir)
    try:
        print(f"   💾 写入向量库：{persist_dir} ...")
        Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    except Exception as e:
        print(f"   ✗ 向量库写入失败：{e}")
        return {'category': category, 'status': 'failed', 'error': str(e)}

    # ── 验证 ──
    validate_vector_db(persist_dir, embeddings)

    # ── 更新哈希 ──
    for f in all_txt_files:
        hash_store[str(f)] = _file_sha256(str(f))

    elapsed = time.time() - t0
    print(f"   ⏱ 耗时 {elapsed:.1f}s  ──  【{category}】完成 ✓")
    return {
        'category':        category,
        'status':          'success',
        'doc_count':       len(all_docs),
        'chunk_count':     len(split_docs),
        'doc_type_counts': doc_type_counts,
        'elapsed_s':       round(elapsed, 1),
    }


def process_all_categories(
    force_rebuild: bool = False,
    categories: list = None,
    parallel: bool = False,
    skip_cases: bool = False,
):
    """
    主入口（v2.0）：
    1. 处理所有（或指定的）分类向量库
    2. ★ 新增：构建独立案例库（chroma_db/典型案例）
    3. ★ 新增：生成/更新知识库版本清单（kb_manifest.json）
    """
    if not os.path.exists(SOURCE_LAWS_DIR):
        raise FileNotFoundError(
            f"法律条文目录不存在：{SOURCE_LAWS_DIR}\n"
            "请确认已将法律文本放置于正确路径。"
        )

    target_categories = categories or LAW_CATEGORIES
    t_total = time.time()

    print("\n" + "="*60)
    print("ContractClarity  法律向量库构建（动态知识库版 v2.0）")
    print("="*60)
    print(f"目标分类：{len(target_categories)} 个")
    print(f"强制重建：{'是' if force_rebuild else '否（增量）'}")
    print(f"并行模式：{'是（线程数 %d）' % MAX_WORKERS if parallel else '否'}")
    print(f"案例库：{'跳过' if skip_cases else '包含'}")
    print()

    # ── 初始化嵌入模型 ──
    print("⚙  初始化嵌入模型 BAAI/bge-large-zh-v1.5 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    print("✓  嵌入模型就绪\n")

    hash_store = _load_hash_store()
    hash_lock  = threading.Lock()
    results    = []

    # ── 处理分类库 ──
    if parallel and len(target_categories) > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_map = {
                ex.submit(process_category, cat, embeddings, hash_store, force_rebuild): cat
                for cat in target_categories
            }
            for future in as_completed(future_map):
                try:
                    res = future.result()
                    results.append(res)
                    with hash_lock:
                        _save_hash_store(hash_store)
                except Exception as e:
                    cat = future_map[future]
                    print(f"\n✗ 分类 [{cat}] 线程异常：{e}")
                    results.append({'category': cat, 'status': 'failed', 'error': str(e)})
    else:
        for cat in target_categories:
            try:
                res = process_category(cat, embeddings, hash_store, force_rebuild)
                results.append(res)
                _save_hash_store(hash_store)
            except Exception as e:
                print(f"\n✗ 分类 [{cat}] 处理异常：{e}")
                results.append({'category': cat, 'status': 'failed', 'error': str(e)})

    # ★ 构建独立案例库
    if not skip_cases:
        try:
            case_result = process_cases_db(embeddings, hash_store, force_rebuild)
            results.append(case_result)
            _save_hash_store(hash_store)
        except Exception as e:
            print(f"\n✗ 案例库构建异常：{e}")
            results.append({'category': CASES_DB_NAME, 'status': 'failed', 'error': str(e)})

    # ★ 更新知识库版本清单
    _update_kb_manifest(results)

    # ── 汇总报告 ──
    elapsed_total = time.time() - t_total
    print("\n" + "="*60)
    print("构建汇总报告")
    print("="*60)
    success  = [r for r in results if r['status'] == 'success']
    skipped  = [r for r in results if r['status'].startswith('skipped')]
    failed   = [r for r in results if r['status'] == 'failed']

    for r in results:
        icon  = {'success': '✅', 'skipped': '⏭ ', 'skipped_incremental': '⏭ ', 'failed': '❌'}.get(r['status'], '?')
        extra = ''
        if r.get('chunk_count'):
            type_str = ''
            if r.get('doc_type_counts'):
                type_str = '  |  ' + '  '.join(
                    f"{dtype[:2]}: {cnt}"
                    for dtype, cnt in r['doc_type_counts'].items()
                )
            extra = f"  →  {r['doc_count']} 文档 / {r['chunk_count']} 块 / {r.get('elapsed_s','?')}s{type_str}"
        print(f"  {icon}  {r['category']:<14}  [{r['status']}]{extra}")

    print()
    print(f"  成功 {len(success)} 个  |  跳过 {len(skipped)} 个  |  失败 {len(failed)} 个")
    print(f"  总耗时 {elapsed_total:.1f}s")
    print("="*60)

    if failed:
        print(f"\n⚠ 有 {len(failed)} 个分类/库构建失败，请检查日志。")

    return results


def _update_kb_manifest(results: list):
    """★ 根据构建结果更新知识库版本清单"""
    manifest = _load_kb_manifest()
    manifest['schema_version'] = KB_SCHEMA_VERSION
    manifest['last_updated']   = datetime.now().isoformat()

    total_laws   = 0
    total_chunks = 0
    cat_stats    = manifest.get('category_stats', {})

    for r in results:
        if r['status'] == 'success':
            cat = r['category']
            total_laws   += r.get('doc_count', 0)
            total_chunks += r.get('chunk_count', 0)
            cat_stats[cat] = {
                'doc_count':       r.get('doc_count', 0),
                'chunk_count':     r.get('chunk_count', 0),
                'doc_type_counts': r.get('doc_type_counts', {}),
                'last_built':      datetime.now().isoformat(),
            }

    manifest['total_laws']      = total_laws
    manifest['total_chunks']    = total_chunks
    manifest['category_stats']  = cat_stats
    manifest['includes_cases']  = any(r['category'] == CASES_DB_NAME and r['status'] == 'success'
                                      for r in results)

    _save_kb_manifest(manifest)
    print(f"\n📋 知识库清单：{total_laws} 份法律文件 / {total_chunks} 个文本块")
    if manifest['includes_cases']:
        print(f"   ⚖ 案例库：已构建（可用于类案比较分析）")


# ─────────────────── CLI 入口 ───────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ContractClarity 法律向量库构建工具（动态知识库版 v2.0）"
    )
    parser.add_argument(
        '--force', action='store_true',
        help='强制全量重建（忽略哈希缓存）'
    )
    parser.add_argument(
        '--categories', nargs='+',
        help='仅处理指定分类（默认全部）'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help=f'启用多线程并行（最多 {MAX_WORKERS} 线程）'
    )
    parser.add_argument(
        '--skip-cases', action='store_true',
        help='跳过独立案例库构建'
    )
    parser.add_argument(
        '--show-manifest', action='store_true',
        help='展示当前知识库版本清单并退出'
    )
    args = parser.parse_args()

    if args.show_manifest:
        manifest = _load_kb_manifest()
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        exit(0)

    try:
        process_all_categories(
            force_rebuild=args.force,
            categories=args.categories,
            parallel=args.parallel,
            skip_cases=args.skip_cases,
        )
    except Exception as e:
        print(f"\n✗ 程序执行失败：{e}")
        exit(1)
