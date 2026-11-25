# Raw run archive

The complete raw run outputs under `results/runs/` are preserved in the
adjacent archive parts. They include smoke runs, the four-video absolute-gate
pilot, the relative-gate pilot, predictions, update traces, diagnostics, and
evaluation outputs.

Reassemble and extract from the repository root:

```bash
cat results/archives/shift_rmot_runs_20261021.tar.gz.part-* > /tmp/shift_rmot_runs.tar.gz
tar -xzf /tmp/shift_rmot_runs.tar.gz
```

The tracked summary tables and analysis remain directly readable in
`results/shift_rmot_pilot/`; this archive preserves the raw provenance behind
those summaries.

Archive manifest: 913 files, 16,622,583 bytes after gzip. The reassembled
archive SHA-256 is
`ff8344fd3baf517b387938d03662c7df0a4caf8b73138fef60333460338ea575`.
