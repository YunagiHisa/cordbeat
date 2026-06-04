# LTS Policy

This is a draft policy for future stable CordBeat releases. This branch does
not declare an LTS line.

## Support Window

- `main`: active development and security fixes.
- Latest stable tag, once declared: security and critical bug fixes.
- Pre-stable snapshots: unsupported except for migration guidance.

## Backport Scope

Backports are reserved for issues that affect user safety, data integrity,
startup reliability, or documented stable APIs. Feature work should target
`main`.
