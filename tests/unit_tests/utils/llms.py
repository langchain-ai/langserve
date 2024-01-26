"""Fake Chat Model wrapper for testing purposes."""
import asyncio
import re
import time
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Mapping,
    Optional,
    cast,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor


class GenericFakeChatModel(BaseChatModel):
    """A generic fake chat model that can be used to test the chat model interface.

    * Chat model should be usable in both sync and async tests
    * Invokes on_llm_new_token to allow for testing of callback related code for new
      tokens.
    * Includes logic to break messages into message chunk to facilitate testing of
      streaming.
    """

    messages: Iterator[AIMessage]
    """Get an iterator over messages.

    This can be expanded to accept other types like Callables / dicts / strings
    to make the interface more generic if needed.

    Note: if you want to pass a list, you can use `iter` to convert it to an iterator.

    Please note that streaming is not implemented yet. We should try to implement it
    in the future by delegating to invoke and then breaking the resulting output
    into message chunks.
    """
    delay_per_token: Optional[float] = None
    """Delay per token in seconds.
    
    If specified, we will sleep for this amount of time after each token when
    streaming.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        message = next(self.messages)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._stream_implementation(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if self.delay_per_token:
                time.sleep(self.delay_per_token)
            yield chunk

    def _stream_implementation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        chat_result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if not isinstance(chat_result, ChatResult):
            raise ValueError(
                f"Expected generate to return a ChatResult, "
                f"but got {type(chat_result)} instead."
            )

        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            raise ValueError(
                f"Expected invoke to return an AIMessage, "
                f"but got {type(message)} instead."
            )

        content = message.content

        if content:
            # Use a regular expression to split on whitespace with a capture group
            # so that we can preserve the whitespace in the output.
            assert isinstance(content, str)
            content_chunks = cast(List[str], re.split(r"(\s)", content))

            for token in content_chunks:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)

        if message.additional_kwargs:
            for key, value in message.additional_kwargs.items():
                # We should further break down the additional kwargs into chunks
                # Special case for function call
                if key == "function_call":
                    for fkey, fvalue in value.items():
                        if isinstance(fvalue, str):
                            # Break function call by `,`
                            fvalue_chunks = cast(List[str], re.split(r"(,)", fvalue))
                            for fvalue_chunk in fvalue_chunks:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content="",
                                        additional_kwargs={
                                            "function_call": {fkey: fvalue_chunk}
                                        },
                                    )
                                )
                                yield chunk
                                if run_manager:
                                    run_manager.on_llm_new_token(
                                        "",
                                        chunk=chunk,  # No token for function call
                                    )
                        else:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    additional_kwargs={"function_call": {fkey: fvalue}},
                                )
                            )
                            yield chunk
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    "",
                                    chunk=chunk,  # No token for function call
                                )
                else:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="", additional_kwargs={key: value}
                        )
                    )
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(
                            "",
                            chunk=chunk,  # No token for function call
                        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            if self.delay_per_token:
                await asyncio.sleep(self.delay_per_token)
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"


class FakeListLLM(LLM):
    """Fake LLM for testing purposes."""

    responses: List[str]
    sleep: Optional[float] = None
    i: int = 0

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}
