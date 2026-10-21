"""App settings models — the assistant identity catalog."""

from pydantic import BaseModel, ConfigDict


class AssistantIdentity(BaseModel):
    """One catalog identity. Its ``id`` sets the assistant's voice; use only ``status == "active"``."""

    model_config = ConfigDict(extra="allow")

    id: str
    first_name: str = ""
    last_name: str = ""
    locale: str = ""
    description: str = ""
    greeting_audio_url: str = ""
    status: str = ""


class AssistantIdentityCatalog(BaseModel):
    """The identities visible to the caller, with a ``version`` that changes when they change."""

    model_config = ConfigDict(extra="allow")

    version: str
    identities: list[AssistantIdentity]
