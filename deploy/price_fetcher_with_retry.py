"""
EnerTEF price fetcher — appelle ENTSO-E day-ahead API pour la zone DE-LU
et insère les prix dans contextual.prices.

Trigger: EventBridge cron quotidien 13h30 UTC (après publication day-ahead 12h55 UTC)
"""
import os
import json
import boto3
import psycopg2
from datetime import datetime, timedelta, timezone
from entsoe import EntsoePandasClient
import pandas as pd

PG_HOST = os.environ['PG_HOST']
PG_PORT = os.environ['PG_PORT']
PG_DB = os.environ['PG_DB']
PG_USER = os.environ['PG_USER']
PG_PASSWORD = os.environ['PG_PASSWORD']
SECRET_NAME = os.environ.get('ENTSOE_SECRET_NAME', 'enertef/entsoe-token')
ZONE = 'DE_LU'  # Luxembourg couple with Germany for day-ahead

def get_entsoe_token():
    """Fetch token from Secrets Manager."""
    client = boto3.client('secretsmanager', region_name='eu-central-1')
    response = client.get_secret_value(SecretId=SECRET_NAME)
    return response['SecretString']

# Same reasoning as the weather puller. On 2026-09-24 a 30 s read timeout
# against web-api.tp.entsoe.eu left contextual.prices empty, and the
# controller spent 2026-09-25 on the flat-price fallback where savings are
# exactly zero by construction. One retry would have absorbed it.
RETRY_DELAYS = [30, 120]          # seconds; 3 attempts total


def _with_retry(fn, what):
    """Call fn(), retrying transient failures. Auth errors are not retried."""
    import time
    last = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e)
            if '401' in msg or '403' in msg:
                # a rejected credential will not fix itself in two minutes,
                # and retrying only delays the report
                print(f'[{what}] auth failure, not retrying: {e}')
                raise
            print(f'[{what}] attempt {attempt + 1} failed: '
                  f'{type(e).__name__}: {e}')
        if attempt < len(RETRY_DELAYS):
            d = RETRY_DELAYS[attempt]
            print(f'[{what}] retrying in {d}s')
            time.sleep(d)
    print(f'[{what}] all {len(RETRY_DELAYS) + 1} attempts failed')
    raise last


def fetch_prices(start=None, end=None):
    """Fetch day-ahead prices from ENTSO-E.

    Default: tomorrow, as scheduled. With `start`/`end` (ISO dates, UTC) an
    explicit window instead -- the backfill path. Added 2026-09-22 after the
    token in Secrets Manager was rotated at ENTSO-E but not here, which cost
    six days of prices (401 daily from 2026-09-16); without this there was no
    way to recover them short of hand-editing the database.
    """
    token = get_entsoe_token()
    client = EntsoePandasClient(api_key=token)
    if start is None:
        now_utc = datetime.now(timezone.utc)
        tomorrow_start = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_end = tomorrow_start + timedelta(days=1)
    else:
        tomorrow_start = pd.Timestamp(start, tz='UTC').to_pydatetime()
        tomorrow_end = pd.Timestamp(end, tz='UTC').to_pydatetime()
    start_ts = pd.Timestamp(tomorrow_start).tz_convert('Europe/Brussels')
    end_ts = pd.Timestamp(tomorrow_end).tz_convert('Europe/Brussels')
    print(f'[ENTSOE] Fetching day-ahead prices for {ZONE} from {start_ts} to {end_ts}')
    prices = _with_retry(
        lambda: client.query_day_ahead_prices(ZONE, start=start_ts, end=end_ts),
        'ENTSOE')
    prices_utc = prices.tz_convert('UTC')
    print(f'[ENTSOE] Received {len(prices_utc)} price points')
    print(f'[ENTSOE] Range: min={prices_utc.min():.2f}, max={prices_utc.max():.2f}, avg={prices_utc.mean():.2f} EUR/MWh')
    return prices_utc

def insert_prices(prices):
    """Insert prices into contextual.prices with upsert logic."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )
    cur = conn.cursor()
    inserted = 0
    for ts, price in prices.items():
        cur.execute("""
            INSERT INTO contextual.prices (timestamp, price_eur_mwh, source)
            VALUES (%s, %s, 'entsoe_dayahead')
            ON CONFLICT (timestamp) DO UPDATE
            SET price_eur_mwh = EXCLUDED.price_eur_mwh,
                source = EXCLUDED.source,
                created_at = NOW()
        """, (ts.to_pydatetime(), float(price)))
        inserted += 1
    conn.commit()
    cur.close()
    conn.close()
    print(f'[DB] Inserted/updated {inserted} price rows')
    return inserted

def lambda_handler(event, context):
    try:
        event = event or {}
        prices = fetch_prices(event.get('start'), event.get('end'))
        if prices.empty:
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No prices available yet', 'inserted': 0})
            }
        inserted = insert_prices(prices)
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Success',
                'inserted': inserted,
                'zone': ZONE,
                'min_eur_mwh': float(prices.min()),
                'max_eur_mwh': float(prices.max()),
                'avg_eur_mwh': float(prices.mean())
            })
        }
    except Exception as e:
        print(f'[ERROR] {e}')
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }