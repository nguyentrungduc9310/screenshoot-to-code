# eval/core.py
from config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_RESOURCE_NAME,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)
from llm import (
    Llm,
    stream_claude_response,
    stream_gemini_response,
    stream_openai_response,
)
from prompts import assemble_prompt
from prompts.types import Stack
from openai.types.chat import ChatCompletionMessageParam


async def generate_code_for_image(image_url: str, stack: Stack, model: Llm) -> str:
    """
    Assemble prompt messages từ image_url và stack,
    sau đó gọi hàm generate_code_core để lấy code.
    """
    prompt_messages = assemble_prompt(image_url, stack)
    return await generate_code_core(prompt_messages, model)


async def generate_code_core(
    prompt_messages: list[ChatCompletionMessageParam], model: Llm
) -> str:
    """
    Dựa vào model được chỉ định, gọi provider tương ứng để stream kết quả:
      - Với các model thuộc nhóm Anthropic (Claude): sử dụng stream_claude_response.
      - Với model Gemini: sử dụng stream_gemini_response.
      - Với các model khác (ví dụ: GPT-4 Vision, GPT-4 Turbo, …): 
          Nếu có Azure OpenAI key thì ưu tiên dùng Azure OpenAI,
          nếu không thì dùng OpenAI thông thường.
    
    Hàm process_chunk là placeholder để xử lý từng đoạn stream nếu cần.
    """
    async def process_chunk(chunk: str):
        # Placeholder: Bạn có thể xử lý từng đoạn chunk (ví dụ: ghi log hay cập nhật UI) nếu cần.
        pass

    if model in (
        Llm.CLAUDE_3_SONNET,
        Llm.CLAUDE_3_5_SONNET_2024_06_20,
        Llm.CLAUDE_3_5_SONNET_2024_10_22,
    ):
        if not ANTHROPIC_API_KEY:
            raise Exception("Anthropic API key not found")
        completion = await stream_claude_response(
            prompt_messages,
            api_key=ANTHROPIC_API_KEY,
            callback=lambda x: process_chunk(x),
            model=model,
        )
    elif model == Llm.GEMINI_2_0_FLASH_EXP:
        if not GEMINI_API_KEY:
            raise Exception("Gemini API key not found")
        completion = await stream_gemini_response(
            prompt_messages,
            api_key=GEMINI_API_KEY,
            callback=lambda x: process_chunk(x),
            model=model,
        )
    else:
        # Với các model dựa trên OpenAI, ưu tiên dùng Azure nếu có cấu hình đầy đủ
        if AZURE_OPENAI_API_KEY:
            completion = await stream_openai_response(
                prompt_messages,
                api_key=AZURE_OPENAI_API_KEY,
                base_url=None,  # Với Azure, endpoint được cấu hình qua resource name & deployment
                callback=lambda x: process_chunk(x),
                model=model,
                azure_api_version=AZURE_OPENAI_API_VERSION,
                resource_name=AZURE_OPENAI_RESOURCE_NAME,
                deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            )
        elif OPENAI_API_KEY:
            completion = await stream_openai_response(
                prompt_messages,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                callback=lambda x: process_chunk(x),
                model=model,
            )
        else:
            raise Exception("No Azure OpenAI API key or OpenAI API key found")
    return completion