import io
import json
import mimetypes
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import markdownify
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from autogen import Agent, AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, OpenAIWrapper, UserProxyAgent

import time
# Optional PDF support
IS_PDF_CAPABLE = False
try:
    import pdfminer
    import pdfminer.high_level

    IS_PDF_CAPABLE = True
except ModuleNotFoundError:
    pass

# Other optional dependencies
try:
    import pathvalidate
except ModuleNotFoundError:
    pass


class SimpleTextBrowser:
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 1024 * 8,
        downloads_folder: Optional[Union[str, None]] = None,
        bing_base_url: str = "https://www.googleapis.com/customsearch/v1",
        bing_api_key: Optional[Union[str, None]] = None,
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: List[str] = list()
        self.page_title: Optional[str] = None
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.bing_base_url = bing_base_url
        self.bing_api_key = bing_api_key
        self.request_kwargs = request_kwargs

        self._page_content = ""

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1]

    def set_address(self, uri_or_path: str) -> None:
        self.history.append(uri_or_path)

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("chrome:"):
            self._serpapi_search(uri_or_path[len("chrome:") :].strip())
        else:
            if not uri_or_path.startswith("http:") and not uri_or_path.startswith("https:"):
                uri_or_path = urljoin(self.address, uri_or_path)
                self.history[-1] = uri_or_path  # Update the address with the fully-qualified path
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def visit_page(self, path_or_uri: str) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri)
        return self.viewport

    def _split_pages(self) -> None:
        # Split only regular pages
        if not self.address.startswith("http:") and not self.address.startswith("https:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx


    def _serpapi_search(self, query: str, filter_year: Optional[int] = None, searchType: Optional[str] = None) -> None:
        # if self.serpapi_key is None:
        #     raise ValueError("Missing SerpAPI key.")

        args = {
            "q": query,
            "api_key": "AIzaSyCXiSx4lmUsPRYjIjupjsgM3VFqyoaFm48",
            "cse_id": "a44b40e16113a4876",
            "num": 10,
        }

        def _google_search(q, api_key, cse_id, **kwargs):
            print("############# Google Search has been used")
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=q, cx=cse_id, **kwargs).execute()
            return res['items']

        if filter_year is not None:
            args["sort"] = f"date:r:{filter_year}0101:{filter_year}1231"

        if searchType is not None:
            args["searchType"] = "image"

        results = _google_search(**args)

        self.page_title = f"{query} - Search"
        if len(results) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter.")
            return

        def _prev_visit(url):
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: List[str] = list()
        idx = 0
        if len(results) > 0:
            for page in results:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)


        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)

    def _fetch_page(self, url: str) -> None:
        try:
            # Prepare the request parameters
            request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
            request_kwargs["stream"] = True

            # Send a HTTP request to the URL
            response = requests.get(url, **request_kwargs)
            response.raise_for_status()

            # If the HTTP request returns a status code 200, proceed
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                for ct in ["text/html", "text/plain", "application/pdf"]:
                    if ct in content_type.lower():
                        content_type = ct
                        break

                if content_type == "text/html":
                    # Get the content of the response
                    html = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        html += chunk

                    soup = BeautifulSoup(html, "html.parser")

                    # Remove javascript and style blocks
                    for script in soup(["script", "style"]):
                        script.extract()

                    # Convert to markdown -- Wikipedia gets special attention to get a clean version of the page
                    if url.startswith("https://en.wikipedia.org/"):
                        body_elm = soup.find("div", {"id": "mw-content-text"})
                        title_elm = soup.find("span", {"class": "mw-page-title-main"})

                        if body_elm:
                            # What's the title
                            main_title = soup.title.string
                            if title_elm and len(title_elm) > 0:
                                main_title = title_elm.string
                            webpage_text = (
                                "# " + main_title + "\n\n" + markdownify.MarkdownConverter().convert_soup(body_elm)
                            )
                        else:
                            webpage_text = markdownify.MarkdownConverter().convert_soup(soup)
                    else:
                        webpage_text = markdownify.MarkdownConverter().convert_soup(soup)

                    # Convert newlines
                    webpage_text = re.sub(r"\r\n", "\n", webpage_text)

                    # Remove excessive blank lines
                    self.page_title = soup.title.string
                    self._set_page_content(re.sub(r"\n{2,}", "\n\n", webpage_text).strip())
                elif content_type == "text/plain":
                    # Get the content of the response
                    plain_text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        plain_text += chunk

                    self.page_title = None
                    self._set_page_content(plain_text)
                elif IS_PDF_CAPABLE and content_type == "application/pdf":
                    pdf_data = io.BytesIO(response.raw.read())
                    self.page_title = None
                    self._set_page_content(pdfminer.high_level.extract_text(pdf_data))
                elif self.downloads_folder is not None:
                    # Try producing a safe filename
                    fname = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension

                    # Open a file for writing
                    download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Return a page describing what just happened
                    self.page_title = "Download complete."
                    self._set_page_content(f"Downloaded '{url}' to '{download_path}'.")
                else:
                    self.page_title = f"Error - Unsupported Content-Type '{content_type}'"
                    self._set_page_content(self.page_title)
            else:
                self.page_title = "Error"
                self._set_page_content("Failed to retrieve " + url)
        except requests.exceptions.RequestException as e:
            self.page_title = "Error"
            self._set_page_content(str(e))

import copy
import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.oai.openai_utils import filter_config
from autogen.token_count_utils import count_token, get_max_token_limit

logger = logging.getLogger(__name__)

class WebSurferAgent(ConversableAgent):
    """An agent that acts as a basic web surfer that can search the web and visit web pages."""

    DEFAULT_PROMPT = (
        "You are a helpful AI assistant with access to a web browser (via the provided functions). In fact, YOU ARE THE ONLY MEMBER OF YOUR PARTY WITH ACCESS TO A WEB BROWSER, so please help out where you can by performing web searches, navigating pages, and reporting what you find. Today's date is "
        + datetime.now().date().isoformat()
    )

    DEFAULT_DESCRIPTION = "A helpful assistant with access to a web browser. Ask them to perform web searches, open pages, navigate to Wikipedia, answer questions from pages, and or generate summaries."

    def __init__(
        self,
        name: str,
        system_message: Optional[Union[str, List[str]]] = DEFAULT_PROMPT,
        description: Optional[str] = DEFAULT_DESCRIPTION,
        is_termination_msg: Optional[Callable[[Dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        summarizer_llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        browser_config: Optional[Union[Dict, None]] = None,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            description=description,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )

        self._create_summarizer_client(summarizer_llm_config, llm_config)

        if browser_config is None:
            self.browser_config = {
                "bing_api_key": "AIzaSyCXiSx4lmUsPRYjIjupjsgM3VFqyoaFm48",
                "bing_base_url": "https://www.googleapis.com/customsearch/v1",
            }
        else:
            self.browser_config = browser_config

        # Create the browser
        self.browser = SimpleTextBrowser(**self.browser_config)

        inner_llm_config = copy.deepcopy(llm_config)

        # Set up the inner monologue
        self._assistant = AssistantAgent(
            self.name + "_inner_assistant",
            system_message=system_message,  # type: ignore[arg-type]
            llm_config=inner_llm_config,
            is_termination_msg=lambda m: False,
        )

        self._user_proxy = UserProxyAgent(
            self.name + "_inner_user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            default_auto_reply="",
            is_termination_msg=lambda m: False,
        )

        if inner_llm_config not in [None, False]:
            self._register_functions()

        self.register_reply([Agent, None], WebSurferAgent.generate_surfer_reply, remove_other_reply_funcs=True)
        self.register_reply([Agent, None], ConversableAgent.generate_code_execution_reply)
        self.register_reply([Agent, None], ConversableAgent.generate_function_call_reply)
        self.register_reply([Agent, None], ConversableAgent.check_termination_and_human_reply)

    def _create_summarizer_client(self, summarizer_llm_config: Dict[str, Any], llm_config: Dict[str, Any]) -> None:
        if summarizer_llm_config is None:
            if llm_config is None:
                self.summarizer_llm_config = None
            elif llm_config is False:
                self.summarizer_llm_config = False
            else:
                self.summarizer_llm_config = copy.deepcopy(llm_config)
                if "config_list" in self.summarizer_llm_config:
                    preferred_models = filter_config(
                        self.summarizer_llm_config["config_list"],
                        {"model": ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k"]},
                    )
                    if not preferred_models:
                        logger.warning(
                            "The summarizer did not find the preferred model (gpt-3.5-turbo-16k) in the config list. "
                            "Semantic operations on webpages (summarization or Q&A) might be costly or ineffective."
                        )
                    else:
                        self.summarizer_llm_config["config_list"] = preferred_models
        else:
            self.summarizer_llm_config = summarizer_llm_config

        self.summarization_client = (
            None if self.summarizer_llm_config is False else OpenAIWrapper(**self.summarizer_llm_config)
        )

    def _register_functions(self) -> None:
        def _browser_state() -> Tuple[str, str]:
            header = f"Address: {self.browser.address}\n"
            if self.browser.page_title is not None:
                header += f"Title: {self.browser.page_title}\n"

            current_page = self.browser.viewport_current_page
            total_pages = len(self.browser.viewport_pages)

            header += f"Viewport position: Showing page {current_page+1} of {total_pages}.\n"
            return header, self.browser.viewport

        @self._user_proxy.register_for_execution()
        @self._assistant.register_for_llm(
            name="informational_web_search",
            description="Perform an INFORMATIONAL web search query then return the search results.",
        )
        def _informational_search(query: str) -> str:
            self.browser.visit_page(f"chrome: {query}")
            header, content = _browser_state()
            return header.strip() + "\n=======================\n" + content

        @self._user_proxy.register_for_execution()
        @self._assistant.register_for_llm(
            name="navigational_web_search",
            description="Perform a NAVIGATIONAL web search query then immediately navigate to the top result. Useful, for example, to navigate to a particular Wikipedia article or other known destination. Equivalent to Google's \"I'm Feeling Lucky\" button.",
        )
        def _navigational_search(query: str) -> str:
            self.browser.visit_page(f"chrome: {query}")

            m = re.search(r"\[.*?\]\((http.*?)\)", self.browser.page_content)
            if m:
                self.browser.visit_page(m.group(1))

            header, content = _browser_state()
            return header.strip() + "\n=======================\n" + content

        @self._user_proxy.register_for_execution()
        @self._assistant.register_for_llm(
            name="visit_page", description="Visit a webpage at a given URL and return its text."
        )
        def _visit_page(url: str) -> str:
            self.browser.visit_page(url)
            header, content = _browser_state()
            return header.strip() + "\n=======================\n" + content

        @self._user_proxy.register_for_execution()
        @self._assistant.register_for_llm(
            name="page_up",
            description="Scroll the viewport UP one page-length in the current webpage and return the new viewport content.",
        )
        def _page_up() -> str:
            self.browser.page_up()
            header, content = _browser_state()
            return header.strip() + "\n=======================\n" + content

        @self._user_proxy.register_for_execution()
        @self._assistant.register_for_llm(
            name="page_down",
            description="Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content.",
        )
        def _page_down() -> str:
            self.browser.page_down()
            header, content = _browser_state()
            return header.strip() + "\n=======================\n" + content

        if self.summarization_client is not None:
            @self._user_proxy.register_for_execution()
            @self._assistant.register_for_llm(
                name="answer_from_page",
                description="Uses AI to read the page and directly answer a given question based on the content.",
            )
            def _answer_from_page(
                question: Optional[str],
                url: Optional[str] = None,
            ) -> str:
                if url is not None and url != self.browser.address:
                    self.browser.visit_page(url)

                limit = 4096
                try:
                    limit = get_max_token_limit(self.summarizer_llm_config["config_list"][0]["model"])
                except ValueError:
                    pass
                except TypeError:
                    pass

                if limit < 16000:
                    logger.warning(
                        f"The token limit ({limit}) of the WebSurferAgent.summarizer_llm_config, is below the recommended 16k."
                    )

                buffer = ""
                for line in re.split(r"([\r\n]+)", self.browser.page_content):
                    tokens = count_token(buffer + line)
                    if tokens + 1024 > limit:
                        break
                    buffer += line

                buffer = buffer.strip()
                if not buffer:
                    return "Nothing to summarize."

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can summarize long documents to answer question.",
                    }
                ]

                prompt = f"Please summarize the following into one or two paragraph:\n\n{buffer}"
                if question:
                    prompt = f"Please summarize the following into one or two paragraphs with respect to '{question}':\n\n{buffer}"

                messages.append(
                    {"role": "user", "content": prompt},
                )

                response = self.summarization_client.create(context=None, messages=messages)
                extracted_response = self.summarization_client.extract_text_or_completion_object(response)[0]
                return str(extracted_response)

            @self._user_proxy.register_for_execution()
            @self._assistant.register_for_llm(
                name="summarize_page",
                description="Uses AI to summarize the content found at a given url. If the url is not provided, the current page is summarized.",
            )
            def _summarize_page(
                url: Optional[str] = None,
            ) -> str:
                return _answer_from_page(url=url, question=None)

    def generate_surfer_reply(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        if messages is None:
            messages = self._oai_messages[sender]

        self._user_proxy.reset()
        self._assistant.reset()

        self._assistant.chat_messages[self._user_proxy] = list()
        history = messages[0 : len(messages) - 1]
        for message in history:
            self._assistant.chat_messages[self._user_proxy].append(message)

        self._user_proxy.send(
            f"Your browser is currently open to the page '{self.browser.page_title}' at the address '{self.browser.address}'.",
            self._assistant,
            request_reply=False,
            silent=True,
        )

        self._user_proxy.send(messages[-1]["content"], self._assistant, request_reply=True, silent=True)
        agent_reply = self._user_proxy.chat_messages[self._assistant][-1]
        proxy_reply = self._user_proxy.generate_reply(
            messages=self._user_proxy.chat_messages[self._assistant], sender=self._assistant
        )

        if proxy_reply == "":
            return True, None if agent_reply is None else agent_reply["content"]
        else:
            return True, None if proxy_reply is None else proxy_reply["content"]
