import json
import weakref

import aiosqlite

SEEN = weakref.WeakSet()


async def init(db: aiosqlite.Connection):
    await db
    await db.execute(
        "CREATE TABLE IF NOT EXISTS configs (key TEXT PRIMARY KEY, config TEXT)"
    )
    await db.commit()


async def list_configs(db: aiosqlite.Connection):
    if db not in SEEN:
        await init(db)
        SEEN.add(db)
    cursor = await db.execute("SELECT * FROM configs")
    return await cursor.fetchall()


async def get_config(db: aiosqlite.Connection, key: str):
    if not key or not db:
        return None
    if db not in SEEN:
        await init(db)
        SEEN.add(db)
    cursor = await db.execute("SELECT config FROM configs WHERE key=?", (key,))
    row = await cursor.fetchone()
    return json.loads(row[0]) if row else None


async def set_config(db: aiosqlite.Connection, key: str, config: dict):
    if db not in SEEN:
        await init(db)
        SEEN.add(db)
    await db.execute(
        "INSERT OR REPLACE INTO configs VALUES (?, ?)", (key, json.dumps(config))
    )
    await db.commit()
