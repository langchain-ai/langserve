import json
import weakref

import aiosqlite

SEEN = weakref.WeakSet()


async def init(db: aiosqlite.Connection):
    if db in SEEN:
        return

    SEEN.add(db)
    await db
    await db.execute(
        "CREATE TABLE IF NOT EXISTS configs (key TEXT PRIMARY KEY, config TEXT)"
    )
    await db.commit()


async def list_configs(db: aiosqlite.Connection):
    await init(db)
    cursor = await db.execute("SELECT * FROM configs")
    return await cursor.fetchall()


async def get_config(db: aiosqlite.Connection, key: str):
    if not key or not db:
        return None
    await init(db)
    cursor = await db.execute("SELECT config FROM configs WHERE key=?", (key,))
    row = await cursor.fetchone()
    return json.loads(row[0]) if row else None


async def set_config(db: aiosqlite.Connection, key: str, config: dict):
    await init(db)
    await db.execute(
        "INSERT OR REPLACE INTO configs VALUES (?, ?)", (key, json.dumps(config))
    )
    await db.commit()
