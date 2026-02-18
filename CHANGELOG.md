# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-02-16

### Added

- Async SDK client (`AsyncPineAI`) with Socket.IO and HTTP transport layers
- Synchronous wrapper (`PineAI`) for non-async usage
- Authentication flow (email verification with code)
- Session management (create, list, join, leave)
- Real-time chat with async generator streaming
- Task lifecycle support (start, watch, cancel)
- Form handling and payment event support
- Pydantic models for all event and data types
- Stream buffering for text and work-log events
- CLI tool (`pine`) with auth, chat, sessions, and tasks commands
- Full type annotations with `py.typed` marker
- Integration test suite

[0.1.0]: https://github.com/19PINE-AI/pine-python/releases/tag/v0.1.0
