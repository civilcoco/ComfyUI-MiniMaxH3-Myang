# Contributing

Contributions are welcome under GPL-3.0-only.

1. Open an issue describing the bug or proposed behavior.
2. Keep existing workflow widget order backward compatible; append new widgets
   rather than inserting them in the middle.
3. Do not commit model weights, generated media, API credentials, personal
   paths, `__pycache__`, local drafts or third-party assets without a compatible
   redistribution license.
4. Preserve provenance when adapting code. Add the source project, revision,
   copyright and license to `THIRD_PARTY_NOTICES.md`.
5. Run the full regression suite before opening a pull request.

```powershell
pwsh tools\run_tests.ps1 -ComfyRoot "D:\path\to\ComfyUI"
```

If ComfyUI uses a dedicated Python environment, also pass its interpreter with
`-Python "D:\path\to\python.exe"`.

Do not report secrets or a working exploit in a public issue. See
[SECURITY.md](SECURITY.md) first.
