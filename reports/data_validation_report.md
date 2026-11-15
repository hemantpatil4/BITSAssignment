# Data Validation Report

- Student ID: `2025ag05734`
- Validated at: 2026-11-15T13:41:34.153744Z
- Input interactions: **12006**
- Valid interactions: **12000**
- Invalid interactions (quarantined): **6**

## Checks

| Check | Passed | Count/Detail |
|---|---|---|
| users_schema | PASS | [] |
| items_schema | PASS | [] |
| interactions_schema | PASS | [] |
| users_duplicate_ids | PASS | 0 |
| items_duplicate_ids | PASS | 0 |
| interactions_missing_values | FAIL | 1 |
| interactions_duplicates | FAIL | 1 |
| invalid_ratings | FAIL | 1 |
| invalid_timestamps | FAIL | 1 |
| referential_integrity_users | FAIL | 1 |
| referential_integrity_items | FAIL | 1 |

## Error Breakdown

- `missing_values`: 1
- `duplicate_interaction`: 1
- `invalid_rating`: 1
- `invalid_timestamp`: 1
- `unknown_user_id`: 1
- `unknown_item_id`: 1