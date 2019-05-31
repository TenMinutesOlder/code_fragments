from sqlalchemy import create_engine
import pandas as pd


def read_sql(paras, sql):
    try:
        engine = create_engine('mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8' % (paras['user'],
                                                                                  paras['pw'],
                                                                                  paras['host'],
                                                                                  paras['db']))
        conn = engine.connect()
        df = pd.read_sql(sql, conn)
        conn.close()
    except Exception as e:
        print(e)
    return df


def write2db(paras, df, table_name):
    try:
        engine = create_engine('mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8' % (paras['user'],
                                                                                  paras['pw'],
                                                                                  paras['host'],
                                                                                  paras['db']))
        conn = engine.connect()
        df.to_sql(table_name, engine, schema=paras['db'], if_exists='append', index=False)
        conn.close()
    except Exception as e:
        print(e)
    return df

