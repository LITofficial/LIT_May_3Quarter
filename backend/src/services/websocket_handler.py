# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See LICENSE in the project root for license information.
# --------------------------------------------------------------------------------------------

"""WebSocket handling for voice proxy connections using Azure AI VoiceLive SDK."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import simple_websocket.ws  # pyright: ignore[reportMissingTypeStubs]
from azure.ai.voicelive.aio import (
    ConnectionClosed,
    VoiceLiveConnection,
    connect,
)
from azure.ai.voicelive.aio import (
    ConnectionError as VoiceLiveConnectionError,
)
from azure.ai.voicelive.models import (
    AudioEchoCancellation,
    AudioNoiseReduction,
    AvatarConfig,
    AzureSemanticVad,
    AzureStandardVoice,
    Modality,
    RequestSession,
    ServerEventType,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential

from src.config import config
from src.services.managers import AgentManager

logger = logging.getLogger(__name__)

# WebSocket constants
AZURE_VOICE_API_VERSION = "2025-05-01-preview"
AZURE_COGNITIVE_SERVICES_DOMAIN = "cognitiveservices.azure.com"

# Session configuration defaults
DEFAULT_TURN_DETECTION_TYPE = "azure_semantic_vad"
DEFAULT_NOISE_REDUCTION_TYPE = "azure_deep_noise_suppression"
DEFAULT_ECHO_CANCELLATION_TYPE = "server_echo_cancellation"
DEFAULT_AVATAR_CHARACTER = "lisa"
DEFAULT_AVATAR_STYLE = "casual-sitting"
DEFAULT_VOICE_NAME = "en-US-Ava:DragonHDLatestNeural"
DEFAULT_VOICE_TYPE = "azure-standard"

# Message types
SESSION_UPDATE_TYPE = "session.update"
PROXY_CONNECTED_TYPE = "proxy.connected"
ERROR_TYPE = "error"

# Log message truncation length
LOG_MESSAGE_MAX_LENGTH = 100


class VoiceProxyHandler:
    """Handles WebSocket proxy connections between client and Azure Voice API using VoiceLive SDK."""

    def __init__(self, agent_manager: AgentManager):
        """
        Initialize the voice proxy handler.

        Args:
            agent_manager: Agent manager instance
        """
        self.agent_manager = agent_manager

    async def handle_connection(self, client_ws: simple_websocket.ws.Server) -> None:
        """
        Handle a WebSocket connection from a client.

        Args:
            client_ws: The client WebSocket connection
        """
        current_agent_id = None

        try:
            current_agent_id = await self._get_agent_id_from_client(client_ws)
            agent_config = self.agent_manager.get_agent(current_agent_id) if current_agent_id else None

            endpoint = self._build_endpoint()
            credential = self._get_credential()
            model = self._get_model(agent_config)
            query_params = self._build_query_params(current_agent_id, agent_config)

            if not credential:
                await self._send_error(client_ws, "No API key found in configuration")
                return

            # 포인트:
            # 여기서 백엔드가 Azure Voice Live API와 WebSocket 기반 실시간 세션을 엽니다.
            # 브라우저는 이 서버에만 연결하고, 서버가 Azure와의 인증/모델/Agent 연결을
            # 대신 처리하는 프록시 역할을 합니다.
            async with connect(
                endpoint=endpoint,
                credential=credential,
                model=model,
                api_version=AZURE_VOICE_API_VERSION,
                query=query_params,
            ) as azure_conn:
                logger.info("Connected to Azure Voice API via SDK with agent: %s", current_agent_id or "default")

                await self._send_message(
                    client_ws,
                    {"type": PROXY_CONNECTED_TYPE, "message": "Connected to Azure Voice API"},
                )

                await self._send_initial_config(azure_conn, agent_config)
                await self._handle_message_forwarding(client_ws, azure_conn)

        except ConnectionClosed as e:
            logger.info("VoiceLive connection closed: code=%s, reason=%s", e.code, e.reason)
        except VoiceLiveConnectionError as e:
            logger.error("VoiceLive connection error: %s", e)
            await self._send_error(client_ws, str(e))
        except Exception as e:
            logger.error("Proxy error: %s", e)
            await self._send_error(client_ws, str(e))

    async def _get_agent_id_from_client(self, client_ws: simple_websocket.ws.Server) -> Optional[str]:
        """Get agent ID from initial client message."""
        try:
            first_message: str | None = await asyncio.get_event_loop().run_in_executor(
                None,
                client_ws.receive,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            )
            if first_message:
                msg = json.loads(first_message)
                if msg.get("type") == SESSION_UPDATE_TYPE:
                    return msg.get("session", {}).get("agent_id")
        except Exception as e:
            logger.error("Error getting agent ID: %s", e)
        return None

    def _build_endpoint(self) -> str:
        """Build the Azure endpoint URL."""
        resource_name = config["azure_ai_resource_name"]
        return f"https://{resource_name}.{AZURE_COGNITIVE_SERVICES_DOMAIN}"

    def _get_credential(self):
        """Get the Azure credential."""
        api_key = config.get("azure_openai_api_key")
        if api_key:
            return AzureKeyCredential(api_key)
        logger.info("No API key found, using DefaultAzureCredential (managed identity)")
        return AsyncDefaultAzureCredential()

    def _get_model(self, agent_config: Optional[Dict[str, Any]]) -> Optional[str]:
        """Get the model name for the connection."""
        if agent_config and agent_config.get("is_azure_agent"):
            return None
        if agent_config:
            return agent_config.get("model", config["model_deployment_name"])
        if config["agent_id"]:
            return None
        return config["model_deployment_name"]

    def _build_query_params(self, agent_id: Optional[str], agent_config: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Build additional query parameters for the connection."""
        params: Dict[str, str] = {}

        if agent_config and agent_config.get("is_azure_agent"):
            params["agent-id"] = agent_id or ""
            project_name = config["azure_ai_project_name"]
            if project_name:
                params["agent-project-name"] = project_name
        elif not agent_config and config["agent_id"]:
            params["agent-id"] = config["agent_id"]

        return params

    async def _send_initial_config(
        self,
        azure_conn: VoiceLiveConnection,
        agent_config: Optional[Dict[str, Any]],
    ) -> None:
        """Send initial configuration to Azure using SDK typed models."""
        session_config = self._build_session_config(agent_config)
        await azure_conn.session.update(session=session_config)
        logger.debug("Sent initial session configuration via SDK")

    def _build_session_config(self, agent_config: Optional[Dict[str, Any]]) -> RequestSession:
        """Build the session configuration using SDK typed models."""
        voice_name = config.get("azure_voice_name", DEFAULT_VOICE_NAME)
        voice_type = config.get("azure_voice_type", DEFAULT_VOICE_TYPE)

        avatar_character = config.get("azure_avatar_character", DEFAULT_AVATAR_CHARACTER)
        avatar_style = config.get("azure_avatar_style", DEFAULT_AVATAR_STYLE)
        is_photo_avatar = False

        if agent_config and agent_config.get("avatar_config"):
            custom_avatar = agent_config["avatar_config"]
            avatar_character = custom_avatar.get("character", avatar_character)
            avatar_style = custom_avatar.get("style", avatar_style)
            is_photo_avatar = custom_avatar.get("is_photo_avatar", False)

        avatar_config_value = self._build_avatar_config(avatar_character, avatar_style, is_photo_avatar)

        return self._create_request_session(voice_name, voice_type, avatar_config_value, agent_config)

    def _build_avatar_config(self, character: str, style: str, is_photo: bool) -> Any:
        """Build avatar configuration for photo or video avatars."""
        if is_photo:
            return {
                "type": "photo-avatar",
                "model": "vasa-1",
                "character": character,
                "customized": False,
            }
        return AvatarConfig(
            character=character,
            style=style if style else None,
            customized=False,
        )

    def _create_request_session(
        self,
        voice_name: str,
        voice_type: str,
        avatar_config_value: Any,
        agent_config: Optional[Dict[str, Any]],
    ) -> RequestSession:
        """Create the RequestSession with all configuration."""
        session = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO, Modality.AVATAR],
            # Azure Semantic VAD가 사용자의 말이 끝났는지, 잠깐 쉰 것인지 판단합니다.
            # 이 턴 종료 감지가 자연스러운 대화 타이밍의 핵심입니다.
            turn_detection=AzureSemanticVad(type=DEFAULT_TURN_DETECTION_TYPE),
            # Voice Live API가 입력 오디오의 노이즈와 에코를 줄여서,
            # AI가 자기 목소리를 다시 듣거나 주변 소음에 흔들리는 상황을 줄입니다.
            input_audio_noise_reduction=AudioNoiseReduction(type=DEFAULT_NOISE_REDUCTION_TYPE),
            input_audio_echo_cancellation=AudioEchoCancellation(type=DEFAULT_ECHO_CANCELLATION_TYPE),
            # LLM 답변을 바로 음성으로 합성할 목소리를 지정합니다.
            voice=AzureStandardVoice(name=voice_name, type=voice_type),
            # 음성 출력과 동기화되는 아바타 설정입니다.
            avatar=avatar_config_value,
        )

        if agent_config and not agent_config.get("is_azure_agent"):
            # 시나리오별 역할 지시문을 세션에 주입해서 "영어 코치", "면접관",
            # "카페 직원"처럼 상황에 맞는 Agent로 행동하게 합니다.
            session["instructions"] = agent_config.get("instructions")
            session["temperature"] = agent_config.get("temperature")
            session["max_response_output_tokens"] = agent_config.get("max_tokens")

        return session

    async def _handle_message_forwarding(
        self,
        client_ws: simple_websocket.ws.Server,
        azure_conn: VoiceLiveConnection,
    ) -> None:
        """Handle bidirectional message forwarding."""
        # 포인트:
        # 두 방향을 동시에 처리합니다. 사용자의 오디오는 Azure로 계속 보내고,
        # Azure에서 생성되는 전사/LLM/TTS/아바타 이벤트는 즉시 브라우저로 돌려보냅니다.
        # 그래서 요청 하나가 끝나기를 기다리는 HTTP 방식보다 대화 지연이 훨씬 적습니다.
        tasks = [
            asyncio.create_task(self._forward_client_to_azure(client_ws, azure_conn)),
            asyncio.create_task(self._forward_azure_to_client(azure_conn, client_ws)),
        ]

        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _forward_client_to_azure(
        self,
        client_ws: simple_websocket.ws.Server,
        azure_conn: VoiceLiveConnection,
    ) -> None:
        """Forward messages from client to Azure using SDK."""
        try:
            while True:
                message: Optional[Any] = await asyncio.get_event_loop().run_in_executor(
                    None,
                    client_ws.receive,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                )
                if message is None:
                    break

                logger.debug("Client->Azure: %s", str(message)[:LOG_MESSAGE_MAX_LENGTH])

                if isinstance(message, str):
                    parsed = json.loads(message)
                    # 예: input_audio_buffer.append 이벤트에는 브라우저 마이크 오디오 조각이 담깁니다.
                    # 이 작은 조각들이 계속 들어가기 때문에 사용자가 말하는 중에도 처리가 시작됩니다.
                    await azure_conn.send(parsed)
                else:
                    await azure_conn.send(message)

        except ConnectionClosed:
            logger.debug("Azure connection closed during client forwarding")
        except Exception as e:
            logger.debug("Client connection closed during forwarding: %s", e)

    async def _forward_azure_to_client(
        self,
        azure_conn: VoiceLiveConnection,
        client_ws: simple_websocket.ws.Server,
    ) -> None:
        """Forward messages from Azure to client using SDK typed events."""
        try:
            async for event in azure_conn:
                event_dict = event.as_dict() if hasattr(event, "as_dict") else dict(event)
                message = json.dumps(event_dict)
                logger.debug("Azure->Client: %s", message[:LOG_MESSAGE_MAX_LENGTH])

                # 예: 전사 완료, 응답 오디오 delta, 아바타 WebRTC SDP 같은 서버 이벤트를
                # 그대로 브라우저로 전달합니다. 프론트엔드는 이벤트 타입별로 UI/오디오를 갱신합니다.
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    client_ws.send,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                    message,
                )

                if event.type == ServerEventType.ERROR:
                    logger.warning("Azure error event: %s", event_dict)
                elif event.type == ServerEventType.SESSION_CREATED:
                    logger.info("Session created: %s", event_dict.get("session", {}).get("id"))
                elif event.type == ServerEventType.SESSION_UPDATED:
                    logger.info("Session updated")

        except ConnectionClosed as e:
            logger.debug("Azure connection closed: code=%s, reason=%s", e.code, e.reason)
        except Exception as e:
            logger.debug("Error forwarding Azure messages: %s", e)

    async def _send_message(self, ws: simple_websocket.ws.Server, message: Dict[str, str | Dict[str, str]]) -> None:
        """Send a JSON message to a WebSocket."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                ws.send,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                json.dumps(message),
            )
        except Exception:
            pass

    async def _send_error(self, ws: simple_websocket.ws.Server, error_message: str) -> None:
        """Send an error message to a WebSocket."""
        await self._send_message(ws, {"type": ERROR_TYPE, "error": {"message": error_message}})
