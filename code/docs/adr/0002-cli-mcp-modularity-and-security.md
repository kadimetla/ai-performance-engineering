# ADR 0002: CLI/MCP Modularity and Security Scope

## Status
Accepted (2025-12-17)

## Context
Two user-facing surfaces are intentionally broad:

- `cli/aisp.py` implements the unified 10-domain CLI. It is large but provides a single entrypoint with consistent patterns.
- `mcp/mcp_server.py` hosts many MCP tools and runs as JSON-RPC over stdio.

The codebase is primarily an educational + local tooling environment; most workflows assume:
- local execution (developer workstation / CI),
- trusted users, and
- an emphasis on ease of use and discoverability.

## Decision
1. Keep `cli/aisp.py` and `mcp/mcp_server.py` as single entrypoints short-term; prefer internal modularization (separate modules imported into the entrypoint) when refactoring.
2. Treat MCP as a local stdio protocol server. Authentication is out of scope for the default stdio mode; if the server is wrapped for network exposure, it must be deployed behind an authenticated transport (SSH tunnel, reverse proxy with auth, etc.).

## Consequences
- Contributors get a single “source of truth” entrypoint per surface (CLI vs MCP), with predictable discovery and registration.
- If a network-facing MCP transport is introduced, security requirements must be revisited explicitly (authn/authz, rate limiting, auditing).

## Follow-ups (Roadmap)
- Split tool registration into domain modules (bench, gpu, etc.) while preserving the single MCP server entrypoint.
- Split CLI domains into modules (bench, gpu, etc.) while preserving `aisp` UX and global flags.
- If network transport is added: introduce explicit authentication configuration (token/mTLS) and document secure deployment patterns.
