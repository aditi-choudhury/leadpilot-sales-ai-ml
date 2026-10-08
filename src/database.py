import sqlite3
from config.settings import DATABASE,ROOT

def connect():
    DATABASE.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(DATABASE)
    con.execute('PRAGMA foreign_keys=ON')
    con.executescript((ROOT/'sql'/'schema.sql').read_text())
    return con

def store_predictions(scored):
    with connect() as con:
        for _,r in scored.iterrows():
            lead=str(r.get('lead_id',''))
            if not lead:continue
            con.execute('INSERT OR IGNORE INTO leads(lead_id,lead_source,industry) VALUES(?,?,?)',(lead,str(r['lead_source']),str(r['industry'])))
            con.execute('INSERT INTO predictions(lead_id,probability,priority,recommendation) VALUES(?,?,?,?)',(lead,float(r['conversion_probability']),str(r['priority']),str(r['recommended_action'])))
