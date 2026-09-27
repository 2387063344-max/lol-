-- 结构化知识库表结构（SQLite）
--
-- 设计讲究：每张表都带 version 字段。
-- 游戏每两周一个大版本，数值会变；没有版本号就无法判断"这条数据是否过期"，
-- 也无法支撑"当前版本是多少"这类问题，更没法在版本更新后做增量重建。

CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS champions (
    version   TEXT NOT NULL,
    champ_id  TEXT NOT NULL,          -- 英文 id，如 Yasuo
    champ_key TEXT NOT NULL,          -- 数字 key，如 157
    name      TEXT NOT NULL,          -- 中文名，如 疾风剑豪
    title     TEXT,                   -- 称号/本名，如 亚索
    tags      TEXT,                   -- 定位（中文），JSON 数组
    partype   TEXT,                   -- 资源类型：法力值 / 能量 等
    info      TEXT,                   -- 攻击/防御/法术/难度 1-10，JSON
    stats     TEXT,                   -- 基础属性与成长，JSON
    blurb     TEXT,
    lore      TEXT,
    PRIMARY KEY (version, champ_id)
);
CREATE INDEX IF NOT EXISTS idx_champions_name ON champions(name);
CREATE INDEX IF NOT EXISTS idx_champions_title ON champions(title);

CREATE TABLE IF NOT EXISTS champion_skills (
    version     TEXT NOT NULL,
    champ_id    TEXT NOT NULL,
    slot        TEXT NOT NULL,        -- passive / Q / W / E / R
    name        TEXT NOT NULL,
    description TEXT,                 -- 纯文本描述
    tooltip     TEXT,                 -- 含数值的详细说明
    cooldown    TEXT,                 -- 冷却（各级，如 14/12/10/8/6）
    cost        TEXT,                 -- 消耗
    range       TEXT,                 -- 射程
    PRIMARY KEY (version, champ_id, slot)
);

CREATE TABLE IF NOT EXISTS items (
    version     TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    plaintext   TEXT,
    gold_total  INTEGER,
    gold_sell   INTEGER,
    tags        TEXT,
    stats       TEXT,                 -- 属性加成，JSON
    from_items  TEXT,                 -- 合成所需，JSON 数组
    into_items  TEXT,                 -- 可合成，JSON 数组
    PRIMARY KEY (version, item_id)
);
CREATE INDEX IF NOT EXISTS idx_items_name ON items(name);

CREATE TABLE IF NOT EXISTS runes (
    version    TEXT NOT NULL,
    rune_id    INTEGER NOT NULL,
    key_name   TEXT NOT NULL,
    name       TEXT NOT NULL,
    family     TEXT NOT NULL,         -- 主宰 / 精密 / 巫术 / 坚决 / 启迪
    short_desc TEXT,
    long_desc  TEXT,
    PRIMARY KEY (version, rune_id)
);
CREATE INDEX IF NOT EXISTS idx_runes_name ON runes(name);

CREATE TABLE IF NOT EXISTS summoner_spells (
    version     TEXT NOT NULL,
    spell_id    TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    cooldown    TEXT,
    PRIMARY KEY (version, spell_id)
);

-- 记录向量索引构建时使用的嵌入模型，用于一致性校验：
-- 换模型必须重建索引，否则新旧向量混检，结果不可信。
CREATE TABLE IF NOT EXISTS index_meta (
    name       TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
