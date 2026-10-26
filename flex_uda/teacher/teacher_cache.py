"""Cross-worker raw response cache with an atomic live-process lease."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


def request_key(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class TeacherCache:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=60,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS outputs (key TEXT PRIMARY KEY, owner INTEGER, payload TEXT)')

    def get_or_claim(self,key,wait=True):
        while True:
            self.db.execute('BEGIN IMMEDIATE')
            row=self.db.execute('SELECT owner,payload FROM outputs WHERE key=?',(key,)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO outputs VALUES (?,?,NULL)',(key,os.getpid()));self.db.execute('COMMIT')
                return None
            owner,payload=row
            if payload is not None:
                self.db.execute('COMMIT');return json.loads(payload)
            try:os.kill(owner,0);alive=True
            except ProcessLookupError:alive=False
            if owner==os.getpid() or not alive:
                self.db.execute('UPDATE outputs SET owner=? WHERE key=?',(os.getpid(),key));self.db.execute('COMMIT')
                return None
            self.db.execute('COMMIT')
            if not wait:return False
            time.sleep(0.1)

    def put(self,key,payload):
        changed=self.db.execute('UPDATE outputs SET payload=? WHERE key=? AND owner=? AND payload IS NULL',
            (json.dumps(payload,sort_keys=True),key,os.getpid())).rowcount
        if changed!=1:raise RuntimeError('raw teacher cache lease was not owned')

    def get(self,key):
        row=self.db.execute('SELECT payload FROM outputs WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else None

    def stats(self):
        complete,pending=self.db.execute('SELECT SUM(payload IS NOT NULL),SUM(payload IS NULL) FROM outputs').fetchone()
        return dict(complete_raw_responses=complete or 0,pending_leases=pending or 0,path=str(self.path))
