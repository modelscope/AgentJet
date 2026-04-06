## Run the command from openclaw official to install openclaw

curl -fsSL https://openclaw.ai/install.sh | bash

## If you are in container environment, openclaw will not start automatically, run the following command to start gateway

openclaw gateway


## Appendix: openclaw installation details

◇  I understand this is personal-by-default and shared/multi-user use requires lock-down. Continue?
│  Yes
│
◇  Setup mode
│  QuickStart
│
◇  QuickStart ─────────────────────────╮
│                                      │
│  Gateway port: 18789                 │
│  Gateway bind: Loopback (127.0.0.1)  │
│  Gateway auth: Token (default)       │
│  Tailscale exposure: Off             │
│  Direct to chat channels.            │
│                                      │
├──────────────────────────────────────╯
│
◇  Model/auth provider
│  vLLM
│
◇  vLLM base URL
│  http://127.0.0.1:10086/v1
│
◇  vLLM API key
│  sk-ajet
│
◇  vLLM model
│  agentjet-model
│
◇  Model configured ─────────────────────────╮
│                                            │
│  Default model set to vllm/agentjet-model  │
│                                            │
├────────────────────────────────────────────╯
│
◇  Default model
│  Keep current (vllm/agentjet-model)
│
◇  Channel status ───────────────────╮
│                                    │
│  Telegram: needs token             │
│  Discord: needs token              │
│  IRC: needs host + nick            │
│  Slack: needs tokens               │
│  Signal: needs setup               │
│  signal-cli: missing (signal-cli)  │
│  iMessage: needs setup             │
│  imsg: missing (imsg)              │
│  LINE: needs token + secret        │
│  Accounts: 0                       │
│  WhatsApp: not configured          │
│  Google Chat: not configured       │
│  Feishu: installed                 │
│  Google Chat: installed            │
│  Nostr: installed                  │
│  Microsoft Teams: installed        │
│  Mattermost: installed             │
│  Nextcloud Talk: installed         │
│  Matrix: installed                 │
│  BlueBubbles: installed            │
│  Zalo: installed                   │
│  Zalo Personal: installed          │
│  Synology Chat: installed          │
│  Tlon: installed                   │
│  Twitch: installed                 │
│  WhatsApp: installed               │
│                                    │
├────────────────────────────────────╯
│
◇  How channels work ───────────────────────────────────────────────────────────────────────╮
│                                                                                           │
│  DM security: default is pairing; unknown DMs get a pairing code.                         │
│  Approve with: openclaw pairing approve <channel> <code>                                  │
│  Public DMs require dmPolicy="open" + allowFrom=["*"].                                    │
│  Multi-user DMs: run: openclaw config set session.dmScope "per-channel-peer" (or          │
│  "per-account-channel-peer" for multi-account channels) to isolate sessions.              │
│  Docs: channels/pairing              │
│                                                                                           │
│  Telegram: simplest way to get started — register a bot with @BotFather and get going.    │
│  WhatsApp: works with your own number; recommend a separate phone + eSIM.                 │
│  Discord: very well supported right now.                                                  │
│  IRC: classic IRC networks with DM/channel routing and pairing controls.                  │
│  Google Chat: Google Workspace Chat app with HTTP webhook.                                │
│  Slack: supported (Socket Mode).                                                          │
│  Signal: signal-cli linked device; more setup (David Reagans: "Hop on Discord.").         │
│  iMessage: this is still a work in progress.                                              │
│  LINE: LINE Messaging API webhook bot.                                                    │
│  Feishu: 飞书/Lark enterprise messaging with doc/wiki/drive tools.                        │
│  Nostr: Decentralized protocol; encrypted DMs via NIP-04.                                 │
│  Microsoft Teams: Teams SDK; enterprise support.                                          │
│  Mattermost: self-hosted Slack-style chat; install the plugin to enable.                  │
│  Nextcloud Talk: Self-hosted chat via Nextcloud Talk webhook bots.                        │
│  Matrix: open protocol; install the plugin to enable.                                     │
│  BlueBubbles: iMessage via the BlueBubbles mac app + REST API.                            │
│  Zalo: Vietnam-focused messaging platform with Bot API.                                   │
│  Zalo Personal: Zalo personal account via QR code login.                                  │
│  Synology Chat: Connect your Synology NAS Chat to OpenClaw with full agent capabilities.  │
│  Tlon: decentralized messaging on Urbit; install the plugin to enable.                    │
│  Twitch: Twitch chat integration                                                          │
│                                                                                           │
├───────────────────────────────────────────────────────────────────────────────────────────╯
│
◇  Select channel (QuickStart)
│  Skip for now
Updated ~/.openclaw/openclaw.json
Workspace OK: ~/.openclaw/workspace
Sessions OK: ~/.openclaw/agents/main/sessions
│
◇  Web search ─────────────────────────────────────────────────────────────────╮
│                                                                              │
│  Web search lets your agent look things up online.                           │
│  Choose a provider. Some providers need an API key, and some work key-free.  │
│  Docs: https://docs.openclaw.ai/tools/web                                    │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────╯
│
◇  Search provider
│  Skip for now
│
◇  Skills status ─────────────╮
│                             │
│  Eligible: 10               │
│  Missing requirements: 34   │
│  Unsupported on this OS: 7  │
│  Blocked by allowlist: 0    │
│                             │
├─────────────────────────────╯
│
◇  Configure skills now? (recommended)
│  No
│
◇  Hooks ──────────────────────────────────────────────────────────────────╮
│                                                                          │
│  Hooks let you automate actions when agent commands are issued.          │
│  Example: Save session context to memory when you issue /new or /reset.  │
│                                                                          │
│  Learn more: https://docs.openclaw.ai/automation/hooks                   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────╯
│
◇  Enable hooks?
│  Skip for now
Config overwrite: /root/.openclaw/openclaw.json (sha256 faa02d438ff95b1b388fa5421a62f223e5154b612004c8a498dfba81e1e96f84 -> e43dced5c2b7f4f03264924a276eb6521b6327734c96e7a51d6076d65177f0e1, backup=/root/.openclaw/openclaw.json.bak)
│
◇  Systemd ───────────────────────────────────────────────────────────────────────────────╮
│                                                                                         │
│  Systemd user services are unavailable. Skipping lingering checks and service install.  │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────╯
