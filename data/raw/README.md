# data/raw: provenance

Written by `tools/fetch_nerrs.py`. Do not edit by hand; re-run the script.

## Source

NOAA National Estuarine Research Reserve System (NERRS), System-Wide
Monitoring Program (SWMP), Apalachicola Bay, Florida, water-quality
sondes, as redistributed in the SWMPr R package:

- `https://raw.githubusercontent.com/fawda123/SWMPr/master/data/apadbwq.RData` (Dry Bar, main dataset)
- `https://raw.githubusercontent.com/fawda123/SWMPr/master/data/apacpwq.RData` (Cat Point, cross-site test)

## What was done

- Columns used: `datetimestamp`, `temp` (°C), `sal` (ppt/psu), `do_mgl` (mg/L), `ph`.
- A value is kept only if its QC flag (`f_temp`, `f_sal`, `f_do_mgl`, `f_ph`)
  starts with `<0>` (passed NERRS QC); every other value is blank.
- Timestamps shifted from UTC to local standard time (UTC-5, no daylight saving).
- Window kept: 2013-06-01 00:00 to 2013-09-30 23:45, 15-minute readings.

| File | Site | Rows | temp kept | sal kept | do_mgl kept | ph kept |
|---|---|---|---|---|---|---|
| `nerrs_apadb_2013_jun_sep.csv` | Dry Bar | 11,712 | 11,700 | 11,700 | 11,538 | 11,611 |
| `nerrs_apacp_2013_jun_sep.csv` | Cat Point | 11,712 | 11,379 | 11,379 | 11,367 | 11,373 |

## Caveat

This is **estuarine monitoring data**, not a prawn pond: the water is
tidal, and there are no aerators, feeding or stocking. It tests the
pipeline on real sensor behaviour; it does not describe a vannamei pond.

## Citations

- NOAA National Estuarine Research Reserve System (NERRS). System-wide
  Monitoring Program. Data accessed from the NOAA NERRS Centralized Data
  Management Office website: http://www.nerrsdata.org
- Beck, M. W. (2016). SWMPr: An R package for retrieving, organizing, and
  analyzing environmental data for estuaries. The R Journal, 8(1), 219-232.
