#!/usr/bin/env python3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

#from typing import List, Dict, Any, Union, Optional, get_type_hints
from typing import *
from typing import Any

from dotenv import load_dotenv
from loguru import logger
import click
from click.testing import CliRunner
from pydantic import BaseModel
from nicegui import Client, ui

from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.llms import Cohere
from langchain.callbacks.manager import Callbacks

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage

load_dotenv()  # load environment variables from .env file

STYLE = r"""
a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}
.markdown ol {
    list-style-type: decimal;
    list-style-position: inside;
}
.markdown ol li::marker {
    color: grey;
}
.markdown ul {
    list-style-type: disc;
    list-style-position: inside;
}
body.body--dark .q-footer {
    background-image: linear-gradient(
    to bottom,
    var(--q-dark-page) 0%,
    transparent 100%
    );
}
"""

class LLMCaps(BaseModel):
    asyncable = False

LLM_CAPS_MAP = {
    ChatOpenAI: LLMCaps(asyncable=True),
    ChatGooglePalm: LLMCaps(asyncable=True),
    Cohere: LLMCaps(),
}

def llm_capable(llm: LLM, capability: str):
    return getattr(LLM_CAPS_MAP[llm.__class__], capability)

def asyncable(llm: LLM):
    return llm_capable(llm, 'asyncable')

def streamable(llm: LLM):
    return hasattr(llm, 'streaming')

def streaming(llm: LLM):
    return streamable(llm) and llm.streaming

class LLMFactory:
    def __init__(self, name: str, produce: Callable) -> None:
        self.name = name
        self.produce = produce

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.produce()

LLM_FACTORIES = [
    LLMFactory('ChatOpenAI', lambda: ChatOpenAI(callbacks=callbacks, streaming=True)),
    LLMFactory('ChatGooglePalm', lambda: ChatGooglePalm(callbacks=callbacks)),
    LLMFactory('Cohere', lambda: Cohere(model='command-xlarge-nightly', callbacks=callbacks, max_tokens=512)),
]

LLM_FACTORY_MAP = {}
LLM_FACTORY_REVERSE_MAP = {}
for factory in LLM_FACTORIES:
    LLM_FACTORY_MAP[factory.name] = factory
    LLM_FACTORY_REVERSE_MAP[factory] = factory.name

def produce_llm(factory: LLMFactory) -> LLM:
    return factory()

messages: List[BaseMessage] = []
thinking: bool = False


class Options:
    def __init__(self) -> None:
        self._llm_factory: Callable = LLM_FACTORY_MAP["ChatOpenAI"]
        self._log_visible: bool = False

    @property
    def llm_factory(self) -> LLM:
        return self._llm_factory

    @llm_factory.setter
    def llm_factory(self, value: Callable):
        self._llm_factory = value
        create_chain()

    @property
    def log_visible(self):
        return self._log_visible

    @log_visible.setter
    def log_visible(self, value: bool):
        self._log_visible = value

    def toggle_log(self):
        self._log_visible = not self._log_visible


options = Options()


class NiceChainCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        logger.info("Prompts after formatting:")
        for prompt in prompts:
            logger.info(prompt)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        messages[-1].content += token
        chat_messages.refresh()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        logger.info(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        logger.info("Finished chain.")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        logger.info(action.log)

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        logger.info(f"{observation_prefix}{output}")
        logger.info(llm_prefix)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on text."""
        logger.info(text)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        logger.info(finish.log)


class LLMChainExecutor:
    def __init__(self, chain: LLMChain) -> None:
        self.chain = chain

    def _run(self, *args: Any, **kwargs: Any):
        return self.chain.run(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> str:
        loop = asyncio.get_event_loop()
        partial_run = functools.partial(self._run, *args, **kwargs)
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, partial_run)
        return result


class AsyncLLMChainExecutor(LLMChainExecutor):
    async def arun(self, *args: Any, **kwargs: Any) -> str:
        return await self.chain.arun(*args, **kwargs)


callbacks: Callbacks = [NiceChainCallbackHandler()]

llm: LLM = None
memory: ConversationBufferMemory = None
chain: LLMChain = None
executor: LLMChainExecutor = None


@click.group()
def slash():
    pass


@slash.command()
def clear():
    global memory, messages
    memory.clear()
    messages = []
    chat_messages.refresh()

@ui.refreshable
async def llm_properties() -> None:
    hints = get_type_hints(llm, localns=globals())
    for key, type in hints.items():
        value = getattr(llm, key)
        if type == bool:
            ui.switch(key, value=value).bind_value(llm, key)
        elif type == int:
            ui.number(label=key, value=int(value), format='%d').bind_value(llm, key, forward=lambda x: int(x))
        elif type == float:
            ui.number(label=key, value=value, format='%.2f').bind_value(llm, key, forward=lambda x: float(x))
        elif type == str:
            ui.input(label=key, value=value).bind_value(llm, key)

@ui.refreshable
async def chat_messages() -> None:
    for message in messages:
        with ui.row():
            if isinstance(message, HumanMessage):
                img = "face"
            elif isinstance(message, AIMessage):
                img = "smart_toy"
            else:
                img = "settings"

            ui.avatar(img)
            with ui.card().tight() as card:
                with ui.card_section():
                    ui.markdown(message.content).classes("text-lg m-2")
    if thinking:
        ui.spinner("dots", size="md").classes("max-w-3xl mx-auto")
    await ui.run_javascript(
        "window.scrollTo(0, document.body.scrollHeight)", respond=False
    )

def clear_messages():
    global messages
    messages = []
    chat_messages.refresh()

def create_chain():
    global options, llm, memory, chain, executor, messages

    llm = produce_llm(options.llm_factory)

    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory, verbose=True)

    #executor = AsyncLLMChainExecutor(chain) if streamable(llm) else LLMChainExecutor(chain)
    executor = AsyncLLMChainExecutor(chain) if asyncable(llm) else LLMChainExecutor(chain)

    clear_messages()
    llm_properties.refresh()

create_chain()

@ui.page("/")
async def main(client: Client):
    async def send() -> None:
        global thinking
        prompt: str = text.value
        text.value = ""
        if not prompt:
            return
        if prompt[0] == "/":
            runner = CliRunner()
            runner.invoke(slash, prompt[1:].split(" "))
            return

        messages.append(HumanMessage(content=prompt))
        thinking = True
        chat_messages.refresh()

        if streaming(llm):
        #if asyncable(llm):
            messages.append(AIMessage(content=""))
            thinking = False
            response = await executor.arun(prompt)
        else:
            response = await executor.arun(prompt)
            messages.append(AIMessage(content=response))
            thinking = False
        chat_messages.refresh()

    ui.add_head_html(f"<style>{STYLE}</style>")

    await client.connected()

    with ui.column().classes("w-full max-w-3xl mx-auto"):
        await chat_messages()

    with ui.left_drawer(top_corner=True, bottom_corner=True):
        with ui.column().classes("w-full"):

            ui.select(LLM_FACTORY_REVERSE_MAP, label='llm').bind_value(options, 'llm_factory')

            await llm_properties()

            dark = ui.dark_mode()
            dark.enable()

            ui.switch(
                "Dark mode", value=dark.value, on_change=lambda value: dark.toggle()
            )

            ui.switch(
                "Show log",
                value=options.log_visible,
                on_change=lambda value: options.toggle_log(),
            )

    with ui.footer():
        with ui.column().classes("w-full max-w-3xl mx-auto my-6"):
            text = (
                ui.input(placeholder="message")
                .props("dark rounded outlined input-class=mx-3")
                .classes("w-full")
                .on("keydown.enter", send)
            )

            ui.markdown(
                "ai chat program built with [NiceGUI](https://nicegui.io) and [LangChain](https://blog.langchain.dev)"
            ).classes("text-xs self-end mr-8 m-[-1em]")

        log = (
            ui.log(max_lines=1000)
            .classes("w-full h-20")
            .bind_visibility(options, "log_visible")
        )

    def log_info(message: str):
        log.push(message)

    logger.add(log_info, level='INFO')

ui.run()
