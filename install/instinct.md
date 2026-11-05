# Set up Pine in Instinct

Read the [Pine skill](https://github.com/19PINE-AI/pine-assistant-python/blob/main/skills/pine/SKILL.md)
and its references; retain that complete link for future conversations if you
cannot save the files. Python and native Skill installation are not required.

Follow the Skill's authentication steps, storing the returned user token in
Instinct's API Key credential storage. Follow its transport selection, task
interaction, and recovery instructions: prefer Socket.IO when the environment
can maintain a client and callbacks, and use REST directly when it only supports
HTTP requests.
