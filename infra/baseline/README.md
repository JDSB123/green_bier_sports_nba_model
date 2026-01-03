Baseline exports for the NBA resource group. Run from repo root:

```
powershell -File scripts/export_rg_baseline.ps1 -ResourceGroupName <rg-name> [-IncludeRoleAssignments]
```

Outputs are written here with timestamps:
- `<rg>-baseline-YYYYMMDD-HHMMSS.json` – raw resource inventory
- `<rg>-tags-YYYYMMDD-HHMMSS.csv` – flat tag report for quick audit
- `<rg>-role-assignments-YYYYMMDD-HHMMSS.json` – optional role assignments (include switch)

Use these files as the desired-state baseline before applying IaC changes. Check them into source control when you want to freeze a reference snapshot. Delete old snapshots as needed.
