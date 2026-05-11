# Databento Reference

Local clone:

- `../databento-python`

Use it as the source of truth for DBN I/O, decoding, and `DBNStore` behavior.

## Files To Read

| Topic | Location |
|------|----------|
| `DBNStore` API | `../databento-python/databento/common/dbnstore.py` |
| Public exports | `../databento-python/databento/__init__.py` |
| Usage tests | `../databento-python/tests/test_historical_bento.py`, `tests/test_historical_client.py` |

## MBP-10 Position

Databento MBP-10 is already Level 2 limit order book data. The repo should use it directly.

- Use `bid_px_00..09`, `ask_px_00..09`, `bid_sz_00..09`, `ask_sz_00..09`, and count columns as the visible book.
- Do not attempt to convert MBP-10 into MBO. Individual order IDs and exact queue positions are not present in MBP.
- For passive execution simulation, use an estimated queue/fill model.

For MBP/MBO data, `DBNStore.to_df()` indexes on `ts_recv` and leaves exchange time in `ts_event`. The execution simulator should preserve both and make an explicit choice about event-time versus receive-time stepping.
