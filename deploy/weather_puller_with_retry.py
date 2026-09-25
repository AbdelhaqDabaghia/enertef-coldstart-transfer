"""
EnerTEF — Open-Meteo Weather Puller (Lambda hors VPC)
======================================================
Fetches 48h hourly weather forecast from Open-Meteo and writes JSON to S3.
Triggered by EventBridge (daily) or manually.

Location: Dudelange, Luxembourg (49.481, 6.085)
Output: S3 file → triggers enertef-pg-loader which writes to contextual.weather
"""

import json
import os
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone

import boto3


s3 = boto3.client('s3', region_name='eu-central-1')

S3_BUCKET = os.environ.get('S3_BUCKET', 'enertef-tef-datalake')
S3_PREFIX = os.environ.get('S3_PREFIX', 'process-data/weather/open-meteo')

# Dudelange, Luxembourg (Copal site)
# Copal Supermarket exact (from NTUA ECC_MCP_v0.5.ipynb)
LAT = 49.70673
LON = 6.48714
LOCATION = 'copal-luxembourg'


# Transient upstream failures cost a whole day here: this job runs once
# daily and the forecaster needs the weather it fetches. On 2026-09-24
# Open-Meteo answered 503 once and the PV model spent the next day on a
# fallback proxy. Retry what is worth retrying, and nothing else.
RETRY_DELAYS = [30, 120]          # seconds; 3 attempts total
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _fetch_with_retry(url, timeout=30):
    """GET with bounded retry. Raises the last error if all attempts fail."""
    import time
    last = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in RETRYABLE_STATUS:
                # 4xx other than 408/429 will not fix itself; fail loudly now
                print(f'[FETCH] HTTP {e.code} is not transient; not retrying')
                raise
            print(f'[FETCH] attempt {attempt + 1} failed: HTTP {e.code}')
        except Exception as e:                      # URLError, timeout, reset
            last = e
            print(f'[FETCH] attempt {attempt + 1} failed: '
                  f'{type(e).__name__}: {e}')
        if attempt < len(RETRY_DELAYS):
            d = RETRY_DELAYS[attempt]
            print(f'[FETCH] retrying in {d}s')
            time.sleep(d)
    print(f'[FETCH] all {len(RETRY_DELAYS) + 1} attempts failed')
    raise last


def lambda_handler(event, context):
    forecast_days = event.get('forecast_days', 2)

    params = {
        'latitude':  LAT,
        'longitude': LON,
        'hourly':    'temperature_2m,cloud_cover,wind_speed_10m,'
                     'shortwave_radiation,direct_radiation,diffuse_radiation,'
                     'relative_humidity_2m,surface_pressure',
        'forecast_days': forecast_days,
        'timezone':  'UTC',
    }
    url = f'https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}'

    print(f'Fetching {forecast_days}-day forecast from Open-Meteo for '
          f'{LOCATION} ({LAT}, {LON})')

    try:
        raw = _fetch_with_retry(url, timeout=30)

        data = json.loads(raw)
        hourly = data['hourly']
        timestamps = hourly['time']
        n_records = len(timestamps)

        # Build records
        records = []
        for i in range(n_records):
            records.append({
                'timestamp':       timestamps[i] + 'Z',
                'location':        LOCATION,
                'temp_celsius':    hourly['temperature_2m'][i],
                'cloud_pct':       hourly['cloud_cover'][i],
                'wind_speed_10m':  hourly['wind_speed_10m'][i],
                'ghi_w_m2':        hourly['shortwave_radiation'][i],
                'dni_w_m2':        hourly['direct_radiation'][i],
                'dhi_w_m2':        hourly['diffuse_radiation'][i],
                'humidity_pct':    hourly['relative_humidity_2m'][i],
                'pressure_msl':    hourly['surface_pressure'][i],
            })

        print(f'Parsed {len(records)} hourly weather records')

        # Write to S3
        now_iso = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        s3_key = f'{S3_PREFIX}/location={LOCATION}/forecast_{now_iso}.json'

        payload = {
            'location':    LOCATION,
            'latitude':    LAT,
            'longitude':   LON,
            'fetched_at':  datetime.now(timezone.utc).isoformat(),
            'count':       len(records),
            'records':     records,
        }

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(payload).encode('utf-8'),
            ContentType='application/json',
        )

        print(f'Wrote {len(records)} weather records to s3://{S3_BUCKET}/{s3_key}')

        return {
            'statusCode': 200,
            'body': json.dumps({
                'count': len(records),
                'location': LOCATION,
                's3_key': s3_key,
                'first': records[0] if records else None,
                'last': records[-1] if records else None,
            })
        }

    except Exception as e:
        print(f'Fatal error: {type(e).__name__}: {e}')
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'type': type(e).__name__})
        }
