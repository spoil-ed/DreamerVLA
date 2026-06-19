
## Full unit suite
- First run failed on repository hygiene because report contained `/mnt/...`; fixed by replacing with non-local wording.
- Re-run: `475 passed, 5 skipped, 12 warnings in 220.84s`.

## Full e2e suite
- `python -m pytest tests/e2e_tests -q`: 25 passed, 1 skipped, 6 warnings in 281.93s.
